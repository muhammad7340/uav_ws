"""VLM outputs -> world-frame detections in the object memory (FLOOD_VLM_PLAN.md 2.3, 2.4)."""
from __future__ import annotations

import math

import numpy as np

from flood_vlm import geometry as geo
from flood_vlm.memory import ObjectMemory, Person

# GT-free height guesses for the ray-plane intersection (plan 2.3). Look-again
# re-centres the drone over the target, which removes most of the parallax.
ROOF_ABOVE_WATER_M = 2.0
DRY_GROUND_ABOVE_WATER_M = 1.0
ROAD_ABOVE_WATER_M = 0.5


# Height of the bbox centre (torso) above the surface, by posture.
TORSO_ABOVE_SURFACE_M = {'sitting': 0.6, 'standing': 1.0, 'lying': 0.15, 'unsure': 0.6}
MAX_SURFACE_Z_M = 12.0
AT_WATER_LEVEL_M = 0.3      # triangulated support this close to the water = in the water


def plane_z(location_or_kind: str, water: float) -> float:
    return {
        'in_water': water,
        'on_roof': water + ROOF_ABOVE_WATER_M,
        'roof': water + ROOF_ABOVE_WATER_M,
        'on_dry_ground': water + DRY_GROUND_ABOVE_WATER_M,
        'mound': water + DRY_GROUND_ABOVE_WATER_M,
        'road': water + ROAD_ABOVE_WATER_M,
    }.get(location_or_kind, water)


def _valid_bbox(b) -> bool:
    """4 numbers with positive size. Values are clipped to 0-1000 in place."""
    if not (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)):
        return False
    b[:] = [min(1000, max(0, v)) for v in b]
    return b[2] > b[0] and b[3] > b[1]


def rel_to_world(frame, x_rel: float, y_rel: float, z: float) -> np.ndarray | None:
    w, h = frame.size
    u, v = geo.rel_to_pixel(x_rel, y_rel, w, h)
    return geo.project_to_plane(u, v, frame.K, frame.R, frame.t, z)


def bbox_ray(frame, bbox) -> tuple[np.ndarray, np.ndarray]:
    w, h = frame.size
    u, v = geo.rel_to_pixel(*geo.bbox_center_rel(bbox), w, h)
    return frame.t.copy(), geo.pixel_ray_world(u, v, frame.K, frame.R)


def refine_height(person: Person, water: float) -> bool:
    """Two-view triangulation (survey + look-again) of the person's torso.

    Sets person.surface_z (what they stand/sit/lie on) and the world position.
    The plan only guessed roof height as water + 2 m; the triage needs the real
    margin above water, so this measures it from the parallax between views.
    """
    p = geo.triangulate(person.rays)
    if p is None:
        return False
    surf = float(p[2]) - TORSO_ABOVE_SURFACE_M.get(person.posture, 0.6)
    if person.location == 'in_water':
        surf = water
    if not water - 0.5 <= surf <= MAX_SURFACE_Z_M:
        return False
    if person.location != 'in_water' and surf - water < AT_WATER_LEVEL_M:
        # a "roof" or "dry ground" at the water line is the water: trust the geometry
        person.location = 'in_water'
        surf = water
    person.surface_z = surf
    person.xyz_world = [float(p[0]), float(p[1]), surf]
    return True


LOOK_OFFSET_M = 3.0
NEAR_VERTICAL_DEG = 10.0


def look_position(person: Person, look_alt: float, min_above_m: float = 4.0) -> tuple[float, float, float]:
    """Where to hover for a look-again.

    Straight above the estimate, unless every view so far was near-vertical: then
    3 m to the side (along the image's long axis) so the new ray makes a ~30 deg
    angle with the old ones and the roof height can be triangulated.
    """
    x, y, z = person.xyz_world
    hz = max(look_alt, z + min_above_m)
    vertical = all(math.degrees(math.acos(min(1.0, abs(float(d[2]))))) < NEAR_VERTICAL_DEG
                   for _, d in person.rays) if person.rays else False
    if vertical:
        y += LOOK_OFFSET_M
    return x, y, hz


def look_prompt(person: Person, frame) -> str:
    """P_LOOK with the expected image position of `person` (it is off-centre after an offset look)."""
    from flood_vlm.prompts import P_LOOK
    x, y, z = person.xyz_world
    uv = geo.world_to_pixel((x, y, z + TORSO_ABOVE_SURFACE_M['unsure']), frame.K, frame.R, frame.t)
    w, h = frame.size
    pt = [500, 500] if uv is None else [int(min(1000, max(0, c))) for c in geo.pixel_to_rel(uv[0], uv[1], w, h)]
    return P_LOOK.format(point=pt)


def process_survey(out: dict, frame, memory: ObjectMemory, tag: str) -> list[dict]:
    """Project P_SURVEY people and dry areas into the world and merge into memory."""
    dets = []
    for k, d in enumerate(out.get('people', []) or []):
        b = d.get('bbox_2d')
        if not _valid_bbox(b):
            continue
        cx, cy = geo.bbox_center_rel(b)
        z = plane_z(d.get('location', ''), frame.water)
        p = rel_to_world(frame, cx, cy, z)
        if p is None:
            continue
        person = memory.add_person(p, d.get('location', 'on_roof'), d.get('posture', 'unsure'),
                                   float(d.get('confidence', 0.5)))
        person.rays.append(bbox_ray(frame, b))
        memory.save_crop(person, frame.image, b, f'{tag}_{k}')
        dets.append({'person_id': person.id, 'xyz': p.tolist(), **d})
    for d in out.get('dry_areas', []) or []:
        b = d.get('bbox_2d')
        if not _valid_bbox(b):
            continue
        z = plane_z(d.get('kind', 'mound'), frame.water)
        poly = [rel_to_world(frame, x, y, z) for x, y in
                ((b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3]))]
        if any(q is None for q in poly):
            continue
        memory.add_zone(d.get('kind', 'mound'), [q[:2].tolist() for q in poly], z)
    return dets


def process_look(out: dict, frame, person: Person, memory: ObjectMemory, tag: str) -> dict:
    """Update `person` from a P_LOOK answer: refined position, labels, beacon point."""
    res = {'person_present': bool(out.get('person_present', False))}
    if not res['person_present'] or not _valid_bbox(out.get('bbox_2d')):
        return res
    b = out['bbox_2d']
    loc = out.get('location', person.location)
    z = plane_z(loc, frame.water)
    cx, cy = geo.bbox_center_rel(b)
    p = rel_to_world(frame, cx, cy, z)
    if p is None:
        return res
    memory.add_person(p, loc, out.get('posture', 'unsure'), float(out.get('confidence', 0.7)),
                      look=True, target=person)
    person.rays.append(bbox_ray(frame, b))
    if refine_height(person, frame.water):
        z = person.surface_z
    elif loc == 'in_water':
        person.surface_z = frame.water
    memory.save_crop(person, frame.image, b, tag)
    person.surface_dry = out.get('surface_dry')
    bp = out.get('beacon_point_2d')
    if isinstance(bp, (list, tuple)) and len(bp) == 2:
        # beacon goes on the same surface as the person (roof) or on dry ground nearby
        bz = z if loc == 'on_roof' else plane_z('on_dry_ground', frame.water)
        q = rel_to_world(frame, bp[0], bp[1], bz)
        if q is not None:
            person.beacon_xy = [float(q[0]), float(q[1])]
            person.beacon_z = float(bz)
            person.beacon_reason = str(out.get('reason', ''))[:200]
    res.update({'xyz': p.tolist(), 'beacon_xy': person.beacon_xy})
    return res
