"""Object memory: grounded people and dry zones in world coordinates (FLOOD_VLM_PLAN.md 2.4)."""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

LOOK_WEIGHT = 3.0        # look-again observations count 3x when averaging positions


@dataclass
class Person:
    id: str
    xyz_world: list[float]
    location: str
    posture: str
    conf: float
    n_obs: int = 1
    weight: float = 1.0                     # sum of merge weights
    crops: list[str] = field(default_factory=list)
    beacon_xy: list[float] | None = None
    beacon_z: float | None = None
    beacon_reason: str = ''
    surface_dry: bool | None = None
    looked: bool = False                    # confirmed by a look-again
    survey_xyz: list[float] | None = None   # first (survey) estimate, for the loc-error metric
    rays: list = field(default_factory=list)  # [(cam_xyz, unit_dir)] through the bbox centre
    surface_z: float | None = None          # triangulated height of what they stand on
    level: str | None = None
    reason: str = ''
    status: str = 'found'                   # found | rejected | beacon_placed

    def to_brief(self, water: float) -> dict:
        x, y, _ = self.xyz_world
        roof = self.location == 'on_roof' and self.surface_z is not None
        return {'person_id': self.id, 'x': round(x, 1), 'y': round(y, 1),
                'location': self.location, 'posture': self.posture,
                'roof_height_est': round(self.surface_z, 1) if roof else None,
                'height_above_water_est': round(self.surface_z - water, 1) if roof else
                (0.0 if self.location == 'in_water' else None),
                'confidence': round(self.conf, 2), 'confirmed': self.looked}


@dataclass
class DryZone:
    id: str
    kind: str
    polygon: list[list[float]]              # world xy
    z: float
    n_obs: int = 1


class ObjectMemory:
    def __init__(self, merge_radius_m: float = 2.0, run_dir: str | None = None):
        self.merge_radius = merge_radius_m
        self.people: list[Person] = []
        self.zones: list[DryZone] = []
        self.run_dir = run_dir
        self._next = 1
        if run_dir:
            os.makedirs(os.path.join(run_dir, 'crops'), exist_ok=True)

    # ── people ────────────────────────────────────────────────────────────────
    def nearest(self, xy, exclude: str | None = None) -> tuple[Person | None, float]:
        best, bd = None, float('inf')
        for p in self.people:
            if p.status == 'rejected' or p.id == exclude:
                continue
            d = math.hypot(p.xyz_world[0] - xy[0], p.xyz_world[1] - xy[1])
            if d < bd:
                best, bd = p, d
        return best, bd

    def add_person(self, xyz, location: str, posture: str, conf: float,
                   look: bool = False, target: Person | None = None) -> Person:
        """Merge into `target` or the nearest person within merge_radius, else create one."""
        w = conf * (LOOK_WEIGHT if look else 1.0)
        p = target
        if p is None:
            cand, d = self.nearest(xyz)
            if cand is not None and d <= self.merge_radius:
                p = cand
        if p is None:
            p = Person(id=f'P{self._next}', xyz_world=[float(v) for v in xyz], location=location,
                       posture=posture, conf=conf, weight=w, looked=look,
                       survey_xyz=None if look else [float(v) for v in xyz])
            self._next += 1
            self.people.append(p)
            return p
        tot = p.weight + w
        p.xyz_world = [(a * p.weight + b * w) / tot for a, b in zip(p.xyz_world, xyz)]
        p.weight = tot
        p.n_obs += 1
        if look:
            # a close look overrides the survey's labels
            p.location, p.conf, p.looked = location, max(conf, p.conf * 0.5), True
            if posture != 'unsure':
                p.posture = posture
        else:
            p.conf = max(p.conf, conf)
            if p.posture == 'unsure' and posture != 'unsure':
                p.posture = posture
        return p

    def save_crop(self, p: Person, img: np.ndarray, bbox_rel, tag: str) -> str | None:
        if not self.run_dir:
            return None
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox_rel
        pad = 0.5
        bw, bh = (x2 - x1) * w / 1000, (y2 - y1) * h / 1000
        u1 = int(max(0, x1 * w / 1000 - pad * bw - 8))
        v1 = int(max(0, y1 * h / 1000 - pad * bh - 8))
        u2 = int(min(w, x2 * w / 1000 + pad * bw + 8))
        v2 = int(min(h, y2 * h / 1000 + pad * bh + 8))
        if u2 <= u1 or v2 <= v1:
            return None
        path = os.path.join(self.run_dir, 'crops', f'{p.id}_{tag}.jpg')
        cv2.imwrite(path, img[v1:v2, u1:u2])
        p.crops.append(path)
        return path

    def active(self) -> list[Person]:
        return [p for p in self.people if p.status != 'rejected']

    def get(self, pid: str) -> Person | None:
        for p in self.people:
            if p.id == pid:
                return p
        return None

    # ── dry zones ─────────────────────────────────────────────────────────────
    def add_zone(self, kind: str, polygon, z: float) -> DryZone:
        cx = sum(q[0] for q in polygon) / len(polygon)
        cy = sum(q[1] for q in polygon) / len(polygon)
        for zn in self.zones:
            zx = sum(q[0] for q in zn.polygon) / len(zn.polygon)
            zy = sum(q[1] for q in zn.polygon) / len(zn.polygon)
            if zn.kind == kind and math.hypot(cx - zx, cy - zy) < 3.0:
                zn.n_obs += 1
                return zn
        zn = DryZone(id=f'Z{len(self.zones) + 1}', kind=kind,
                     polygon=[[round(a, 2), round(b, 2)] for a, b in polygon], z=z)
        self.zones.append(zn)
        return zn

    # ── export ────────────────────────────────────────────────────────────────
    def to_text(self, water: float) -> str:
        """Compact JSON for the triage prompt."""
        return json.dumps([p.to_brief(water) for p in self.active()], separators=(',', ':'))

    def to_dict(self) -> dict:
        people = []
        for p in self.people:
            d = asdict(p)
            d['rays'] = [[list(map(float, t)), list(map(float, r))] for t, r in p.rays]
            people.append(d)
        return {'people': people, 'zones': [asdict(z) for z in self.zones]}
