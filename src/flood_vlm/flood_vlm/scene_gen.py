#!/usr/bin/env python3
"""Procedural flooded-town generator (FLOOD_VLM_PLAN.md 1.2).

    ros2 run flood_vlm scene_gen --seed 7
    ros2 run flood_vlm scene_gen --seed 7 --out <dir>/flood_town_7.sdf --overview-cam

Writes <out>.sdf and <out>.gt.json (exact positions, heights and dry zones).
Layout: a 4 x 3 street grid of 14 m cells, x in {-21,-7,7,21}, y in {-12,2,16};
roads run between the cells; the staging mound sits at `home` south of the grid.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

from flood_vlm.config import drone_worlds_dir, load_config

FUEL = 'https://fuel.gazebosim.org/1.0/OpenRobotics/models/'

CELL_X = (-21.0, -7.0, 7.0, 21.0)
CELL_Y = (-12.0, 2.0, 16.0)
ROAD_X = (-14.0, 0.0, 14.0)          # north-south streets
ROAD_Y = (-19.0, -5.0, 9.0, 23.0)    # east-west streets
ROAD_W = 4.0
STOREY_M = 3.0
PARAPET_H = 0.4
PARAPET_T = 0.2

HOUSE_COLOURS = {
    'off_white': (0.93, 0.91, 0.86),
    'beige': (0.86, 0.78, 0.63),
    'pale_yellow': (0.95, 0.89, 0.58),
    'light_pink': (0.93, 0.76, 0.76),
    'grey': (0.66, 0.66, 0.64),
}
ROOF_RGB = (0.76, 0.75, 0.71)        # concrete slab
GROUND_RGB = (0.36, 0.40, 0.24)      # brown-green
MOUND_RGB = (0.33, 0.47, 0.22)       # grassy
ROAD_RGB = (0.30, 0.30, 0.31)
WATER_RGB = (0.45, 0.36, 0.22)      # plan colour; specular below gives it a wet sheen

CARS = {'hatchback': 'Hatchback', 'suv': 'SUV', 'pickup': 'Pickup'}


# ───────── small SDF helpers ──────────────────────────────────────────────────
def _f(v: float) -> str:
    return f'{v:.3f}'


def pose(x, y, z, roll=0.0, pitch=0.0, yaw=0.0) -> str:
    return f'<pose>{_f(x)} {_f(y)} {_f(z)} {_f(roll)} {_f(pitch)} {_f(yaw)}</pose>'


def material(rgb, transparency=0.0, emissive=None, specular=0.05) -> str:
    r, g, b = rgb
    sp = specular
    em = ''
    if emissive is not None:
        em = f'<emissive>{emissive[0]} {emissive[1]} {emissive[2]} 1</emissive>'
    return (f'<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse>'
            f'<specular>{sp} {sp} {sp} 1</specular>{em}</material>'
            + (f'<transparency>{transparency}</transparency>' if transparency else ''))


def box_visual(name, x, y, z, sx, sy, sz, rgb, yaw=0.0, collision=True) -> str:
    """Box centred at (x, y, z) in the model frame."""
    geo = f'<geometry><box><size>{_f(sx)} {_f(sy)} {_f(sz)}</size></box></geometry>'
    out = (f'<visual name="{name}_v">{pose(x, y, z, yaw=yaw)}{geo}'
           f'<cast_shadows>false</cast_shadows>{material(rgb)}</visual>')
    if collision:
        out += f'<collision name="{name}_c">{pose(x, y, z, yaw=yaw)}{geo}</collision>'
    return out


def static_model(name, x, y, z, yaw, body) -> str:
    return (f'<model name="{name}"><static>true</static>{pose(x, y, 0.0 + z, yaw=yaw)}'
            f'<link name="link">{body}</link></model>')


def include(name, uri, x, y, z, yaw=0.0) -> str:
    return (f'<include><name>{name}</name><uri>{uri}</uri>'
            f'{pose(x, y, z, yaw=yaw)}<static>true</static></include>')


def rect(cx, cy, sx, sy, yaw=0.0) -> list[list[float]]:
    """Corners of a yawed rectangle, counter-clockwise."""
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        lx, ly = u * sx / 2, v * sy / 2
        pts.append([round(cx + c * lx - s * ly, 3), round(cy + s * lx + c * ly, 3)])
    return pts


def local_to_world(cx, cy, yaw, u, v) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return cx + c * u - s * v, cy + s * u + c * v


# ───────── generator ──────────────────────────────────────────────────────────
class SceneGen:
    def __init__(self, seed: int, cfg: dict):
        self.seed = seed
        self.cfg = cfg
        self.rng = random.Random(seed)
        w = cfg['water']
        self.w_start = float(w['start'])
        self.w_final = float(max([w['start']] + list(w['steps'])))
        self.home = (float(cfg['home']['x']), float(cfg['home']['y']))
        self.models: list[str] = []
        self.gt = {
            'seed': seed,
            'world': cfg['world_name'],
            'water': {'start': self.w_start, 'steps': list(w['steps']),
                      'step_every_s': w['step_every_s']},
            'houses': [], 'people': [], 'dry_zones': [], 'roads': [], 'cars': [],
            'trees': [], 'landmarks': [], 'debris': [], 'home': {},
        }

    # -- static town ----------------------------------------------------------
    def ground(self):
        body = box_visual('ground', 0, 0, -0.05, 80, 80, 0.1, GROUND_RGB)
        self.models.append(static_model('ground', 0, 0, 0, 0, body))

    def houses(self) -> list[tuple[float, float]]:
        """Fill 8-12 of the 12 grid cells with flat-roof houses; return empty cells."""
        cells = [(x, y) for y in CELL_Y for x in CELL_X]
        n = self.rng.randint(8, 12)
        chosen = self.rng.sample(cells, n)
        empty = [c for c in cells if c not in chosen]
        # Guarantee at least one low roof (1 storey, no plinth) and one tall house.
        low_idx = self.rng.randrange(n)
        for i, (gx, gy) in enumerate(sorted(chosen, key=lambda c: (c[1], c[0]))):
            hid = f'H{i + 1}'
            sx = round(self.rng.uniform(6.0, 9.0), 2)
            sy = round(self.rng.uniform(6.0, 9.0), 2)
            yaw = round(self.rng.choice((0.0, math.pi / 2)) + self.rng.uniform(-0.05, 0.05), 3)
            cx = round(gx + self.rng.uniform(-0.5, 0.5), 2)
            cy = round(gy + self.rng.uniform(-0.5, 0.5), 2)
            if i == low_idx:
                storeys, plinth = 1, 0.0
            else:
                storeys = self.rng.choice((1, 1, 2))
                plinth = round(self.rng.choice((0.0, self.rng.uniform(0.6, 1.0))), 2)
            roof_z = round(plinth + storeys * STOREY_M, 2)
            colour = self.rng.choice(sorted(HOUSE_COLOURS))
            tank = self.rng.random() < 0.6
            body = ''
            if plinth > 0:
                body += box_visual('plinth', 0, 0, plinth / 2, sx + 1.0, sy + 1.0, plinth,
                                   (0.55, 0.53, 0.50))
            wall_h = storeys * STOREY_M - 0.1
            body += box_visual('walls', 0, 0, plinth + wall_h / 2, sx, sy, wall_h,
                               HOUSE_COLOURS[colour])
            roof_rgb = tuple(round(0.6 * a + 0.4 * b, 3) for a, b in zip(ROOF_RGB, HOUSE_COLOURS[colour]))
            body += box_visual('roof', 0, 0, roof_z - 0.05, sx, sy, 0.1, roof_rgb)
            pz = roof_z + PARAPET_H / 2
            par_rgb = HOUSE_COLOURS[colour]
            body += box_visual('par_n', 0, sy / 2 - PARAPET_T / 2, pz, sx, PARAPET_T, PARAPET_H, par_rgb)
            body += box_visual('par_s', 0, -sy / 2 + PARAPET_T / 2, pz, sx, PARAPET_T, PARAPET_H, par_rgb)
            body += box_visual('par_e', sx / 2 - PARAPET_T / 2, 0, pz, PARAPET_T, sy, PARAPET_H, par_rgb)
            body += box_visual('par_w', -sx / 2 + PARAPET_T / 2, 0, pz, PARAPET_T, sy, PARAPET_H, par_rgb)
            tank_uv = None
            if tank:
                tank_uv = (self.rng.choice((-1, 1)) * (sx / 2 - 1.0),
                           self.rng.choice((-1, 1)) * (sy / 2 - 1.0))
                body += (f'<visual name="tank_v">{pose(tank_uv[0], tank_uv[1], roof_z + 0.5)}'
                         '<geometry><cylinder><radius>0.5</radius><length>1.0</length></cylinder></geometry>'
                         f'<cast_shadows>false</cast_shadows>{material((0.05, 0.05, 0.05))}</visual>')
            self.models.append(static_model(f'house_{hid}', cx, cy, 0, yaw, body))
            house = {'id': hid, 'x': cx, 'y': cy, 'yaw': yaw, 'sx': sx, 'sy': sy,
                     'roof_z': roof_z, 'plinth_z': plinth, 'storeys': storeys,
                     'colour': colour, 'tank_uv': tank_uv,
                     'polygon': rect(cx, cy, sx, sy, yaw)}
            self.gt['houses'].append(house)
            inner_sx, inner_sy = sx - 2 * PARAPET_T, sy - 2 * PARAPET_T
            self.gt['dry_zones'].append({'id': f'D_{hid}', 'type': 'roof', 'house': hid,
                                         'polygon': rect(cx, cy, inner_sx, inner_sy, yaw),
                                         'top_z': roof_z})
        return empty

    def roads(self):
        """Ground-level streets plus 1-2 raised embankment segments (always dry)."""
        segs = []
        for x in ROAD_X:
            segs.append(('ns', x, 0.0, ROAD_W, 54.0))       # y from -27 to 27
        for y in ROAD_Y:
            segs.append(('ew', 0.0, y, 56.0, ROAD_W))
        n_emb = self.rng.randint(1, 2)
        # Embankments: a 12 m piece of an east-west street, ends on cell centres.
        emb_candidates = [(y, x0) for y in ROAD_Y[1:3] for x0 in (-21.0, -7.0, 7.0)]
        embs = self.rng.sample(emb_candidates, n_emb)
        body = ''
        for i, (kind, cx, cy, sx, sy) in enumerate(segs):
            rid = f'R{i + 1}'
            body += box_visual(rid, cx, cy, 0.02, sx, sy, 0.04, ROAD_RGB, collision=False)
            self.gt['roads'].append({'id': rid, 'kind': kind, 'embankment': False,
                                     'polygon': rect(cx, cy, sx, sy), 'top_z': 0.04})
        emb_top = round(self.w_final + 0.3, 2)
        for j, (y, x0) in enumerate(embs):
            rid = f'E{j + 1}'
            cx, length = x0 + 7.0, 10.0
            body += box_visual(rid, cx, y, emb_top / 2, length, ROAD_W, emb_top, ROAD_RGB)
            # earth shoulders so it reads as an embankment
            body += box_visual(rid + '_sh', cx, y, emb_top / 2 - 0.1, length + 1.0, ROAD_W + 0.6,
                               emb_top - 0.2, (0.42, 0.36, 0.24))
            poly = rect(cx, y, length, ROAD_W)
            self.gt['roads'].append({'id': rid, 'kind': 'ew', 'embankment': True,
                                     'polygon': poly, 'top_z': emb_top})
            self.gt['dry_zones'].append({'id': f'D_{rid}', 'type': 'road', 'polygon': poly,
                                         'top_z': emb_top})
        self.models.append(static_model('roads', 0, 0, 0, 0, body))

    def mounds(self, empty: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Staging mound at home + 1-2 more dry mounds in empty cells."""
        top = round(self.w_final + 0.4, 2)
        hx, hy = self.home
        sx, sy = 12.0, 8.0
        body = box_visual('staging', 0, 0, top / 2, sx, sy, top, MOUND_RGB)
        self.models.append(static_model('mound_staging', hx, hy, 0, 0, body))
        self.gt['dry_zones'].append({'id': 'D_M0', 'type': 'mound', 'staging': True,
                                     'polygon': rect(hx, hy, sx, sy), 'top_z': top})
        self.gt['home'] = {'x': hx, 'y': hy, 'z': top}
        self.models.append(include('ambulance', FUEL + 'Ambulance', hx - 3.0, hy - 1.0, top, 0.0))
        self.models.append(include('rc_boat', FUEL + 'RC Boat', hx + 4.0, hy + 4.6, self.w_start, math.pi / 2))
        for k, dx in enumerate((-4.5, -1.5, 1.5)):
            self.models.append(include(f'sandbags_home_{k}', 'model://flood_sandbags',
                                       hx + dx, hy + sy / 2 - 0.4, top))
        n = min(len(empty), self.rng.randint(1, 2))
        picked = self.rng.sample(empty, n)
        for i, (gx, gy) in enumerate(picked):
            cx = gx + self.rng.uniform(-1, 1)
            cy = gy + self.rng.uniform(-1, 1)
            body = box_visual('mound', 0, 0, top / 2, 6.0, 6.0, top, MOUND_RGB)
            self.models.append(static_model(f'mound_{i + 1}', cx, cy, 0, 0, body))
            self.gt['dry_zones'].append({'id': f'D_M{i + 1}', 'type': 'mound',
                                         'polygon': rect(cx, cy, 6.0, 6.0), 'top_z': top})
        return [c for c in empty if c not in picked]

    def water(self):
        """Visual-only muddy water slab; water_controller moves it with set_pose."""
        z = self.w_start - 0.025
        self.models.append(
            '<model name="flood_water"><static>true</static>'
            f'{pose(0, 0, z)}<link name="link"><visual name="water_v">'
            '<geometry><box><size>80 80 0.05</size></box></geometry>'
            f'<cast_shadows>false</cast_shadows>{material(WATER_RGB, transparency=0.15, specular=0.4)}'
            '</visual></link></model>')

    # -- dressing -------------------------------------------------------------
    def _blocked(self, x: float, y: float, margin: float) -> bool:
        """True if (x, y) is within `margin` of a house, dry zone or placed object."""
        from flood_vlm.gt import point_in_polygon
        for poly in [h['polygon'] for h in self.gt['houses']] + [z['polygon'] for z in self.gt['dry_zones']]:
            for dx, dy in ((0, 0), (margin, 0), (-margin, 0), (0, margin), (0, -margin)):
                if point_in_polygon(x + dx, y + dy, poly):
                    return True
        for o in self.gt['cars'] + self.gt['people']:
            if math.hypot(x - o['x'], y - o['y']) < 2.5:
                return True
        return False

    def cars(self) -> list[dict]:
        n = self.rng.randint(3, 5)
        out = []
        for i in range(n):
            # on a ground-level north-south street, between houses, off the embankments
            for _ in range(100):
                x = self.rng.choice(ROAD_X) + self.rng.choice((-1.0, 1.0))
                y = self.rng.choice((-12.0, 2.0, 16.0)) + self.rng.uniform(-3, 3)
                if not self._blocked(x, y, 2.5):
                    break
            kind = self.rng.choice(sorted(CARS))
            yaw = math.pi / 2 + self.rng.uniform(-0.3, 0.3)
            cid = f'C{i + 1}'
            self.models.append(include(f'car_{cid}', FUEL + CARS[kind], x, y, 0.0, yaw))
            car = {'id': cid, 'model': kind, 'x': round(x, 2), 'y': round(y, 2), 'yaw': round(yaw, 3)}
            self.gt['cars'].append(car)
            out.append(car)
        return out

    def trees_and_landmarks(self, empty: list[tuple[float, float]]):
        spots = []
        for gx, gy in empty:                      # leftover cells get a cluster of trees
            for _ in range(2):
                spots.append((gx + self.rng.uniform(-4, 4), gy + self.rng.uniform(-4, 4)))
        for x in (-29.0, 29.0):                   # east and west margins
            for y in (-8.0, 6.0, 20.0):
                spots.append((x + self.rng.uniform(-1, 1), y + self.rng.uniform(-3, 3)))
        self.rng.shuffle(spots)
        n = self.rng.randint(6, 10)
        for i, (x, y) in enumerate(spots[:n]):
            kind = self.rng.choice(('Oak tree', 'Pine Tree'))
            tid = f'T{i + 1}'
            self.models.append(include(f'tree_{tid}', FUEL + kind, x, y, 0.0, self.rng.uniform(0, 6.28)))
            self.gt['trees'].append({'id': tid, 'model': kind, 'x': round(x, 2), 'y': round(y, 2)})
        hx, hy = self.home
        x, y = hx + 25.0, hy - 1.0
        self.models.append(include('water_tower', FUEL + 'Water tower', x, y, 0.0))
        self.gt['landmarks'].append({'id': 'water_tower', 'model': 'Water tower', 'x': x, 'y': y})
        # Gas station built from boxes: the Fuel 'Gas Station' uses OGRE1 material
        # scripts and renders black under ogre2.
        x, y = hx - 20.0, hy + 1.0
        body = box_visual('shop', -3.5, 0, 1.6, 4.0, 6.0, 3.2, (0.95, 0.95, 0.93))
        body += box_visual('stripe', -3.5, 0, 3.0, 4.05, 6.05, 0.3, (0.80, 0.10, 0.08))
        body += box_visual('canopy', 3.0, 0, 4.6, 7.0, 8.0, 0.4, (0.85, 0.12, 0.10))
        for k, (px, py) in enumerate(((0.5, -3), (0.5, 3), (5.5, -3), (5.5, 3))):
            body += box_visual(f'post{k}', px, py, 2.2, 0.3, 0.3, 4.4, (0.7, 0.7, 0.7))
        for k, py in enumerate((-1.5, 1.5)):
            body += box_visual(f'pump{k}', 3.0, py, 0.7, 0.6, 1.0, 1.4, (0.9, 0.85, 0.2))
        self.models.append(static_model('gas_station', x, y, 0, 0, body))
        self.gt['landmarks'].append({'id': 'gas_station', 'model': 'procedural', 'x': x, 'y': y})

    def debris(self):
        for i in range(4):
            x = self.rng.choice(ROAD_X) + self.rng.uniform(-1.2, 1.2)
            y = self.rng.choice(ROAD_Y[1:]) + self.rng.uniform(-1.2, 1.2)
            name = f'debris_{i + 1}'
            yaw = round(self.rng.uniform(0, 6.28), 3)
            self.models.append(include(name, 'model://flood_driftwood', x, y, self.w_start - 0.2, yaw))
            self.gt['debris'].append({'id': name, 'x': round(x, 2), 'y': round(y, 2), 'yaw': yaw})

    # -- people ---------------------------------------------------------------
    def _roof_spot(self, house: dict, avoid: list[tuple[float, float]]) -> tuple[float, float] | None:
        mu, mv = house['sx'] / 2 - 1.2, house['sy'] / 2 - 1.2
        for _ in range(50):
            u, v = self.rng.uniform(-mu, mu), self.rng.uniform(-mv, mv)
            if house['tank_uv'] and math.hypot(u - house['tank_uv'][0], v - house['tank_uv'][1]) < 1.5:
                continue
            x, y = local_to_world(house['x'], house['y'], house['yaw'], u, v)
            if all(math.hypot(x - ax, y - ay) > 2.5 for ax, ay in avoid):
                return x, y
        return None

    def _actor(self, name: str, x: float, y: float, feet_z: float, yaw: float) -> str:
        """Standing person: walk.dae actor that 'walks' in place (reads as wading/waving)."""
        z = feet_z + 1.0          # walk.dae origin is at the hips, ~1 m above the feet
        p = f'{_f(x)} {_f(y)} {_f(z)} 0 0 {_f(yaw)}'
        return (f'<actor name="{name}"><pose>0 0 0 0 0 0</pose>'
                '<skin><filename>model://flood_people/meshes/walk.dae</filename><scale>1.0</scale></skin>'
                '<animation name="walk"><filename>model://flood_people/meshes/walk.dae</filename>'
                '<interpolate_x>false</interpolate_x></animation>'
                '<script><loop>true</loop><delay_start>0</delay_start><auto_start>true</auto_start>'
                f'<trajectory id="0" type="walk"><waypoint><time>0</time><pose>{p}</pose></waypoint>'
                f'<waypoint><time>2</time><pose>{p}</pose></waypoint></trajectory></script></actor>')

    def people(self, cars: list[dict]):
        n = self.rng.randint(6, 9)
        n_lying = self.rng.randint(1, 2)
        n_water = self.rng.randint(1, 2)
        n_roof = n - n_lying - n_water
        houses = sorted(self.gt['houses'], key=lambda h: h['roof_z'])
        used: list[tuple[float, float]] = []
        plan = ['lying'] * n_lying + ['roof'] * n_roof
        self.rng.shuffle(plan)
        # first roof person goes on the lowest roof so a water rise changes triage
        roof_houses = [houses[0]] + self.rng.sample(houses, len(houses))
        k = 0
        pid = 0
        for kind in plan:
            spot = None
            while spot is None and k < len(roof_houses):
                house = roof_houses[k]
                k += 1
                if sum(1 for p in self.gt['people'] if p['house'] == house['id']) >= 2:
                    continue
                spot = self._roof_spot(house, used)
            if spot is None:
                break
            pid += 1
            x, y = spot
            z = house['roof_z']
            yaw = self.rng.uniform(0, 6.28)
            if kind == 'lying':
                model, posture = 'rescue_randy', 'lying'
                self.models.append(include(f'person_P{pid}', FUEL + 'Rescue Randy', x, y, z, yaw))
            elif self.rng.random() < 0.55:
                model, posture = 'rescue_randy_sitting', 'sitting'
                self.models.append(include(f'person_P{pid}', FUEL + 'Rescue Randy Sitting', x, y, z, yaw))
            else:
                model, posture = 'walk_actor', 'standing'
                self.models.append(self._actor(f'person_P{pid}', x, y, z, yaw))
            used.append((x, y))
            self.gt['people'].append({'id': f'P{pid}', 'x': round(x, 2), 'y': round(y, 2),
                                      'z': round(z, 2), 'model': model, 'support': 'roof',
                                      'house': house['id'], 'posture': posture,
                                      'yaw': round(yaw, 3)})
        # people in the water: stand on a submerged low wall next to a car
        walls = ''
        for i in range(n_water):
            car = cars[i % len(cars)]
            # cars sit along a north-south street, so step ahead of / behind the car
            for _ in range(100):
                x = car['x'] + self.rng.uniform(-0.6, 0.6)
                y = car['y'] + self.rng.choice((-1.0, 1.0)) * self.rng.uniform(3.2, 4.5)
                if not self._blocked(x, y, 1.0) and all(math.hypot(x - ax, y - ay) > 2.5 for ax, ay in used):
                    break
            feet = round(self.rng.uniform(0.6, 0.9), 2)
            walls += box_visual(f'wall{i}', x, y, feet / 2, 1.0, 1.6, feet, (0.47, 0.30, 0.25))
            pid += 1
            yaw = self.rng.uniform(0, 6.28)
            self.models.append(self._actor(f'person_P{pid}', x, y, feet, yaw))
            used.append((x, y))
            self.gt['people'].append({'id': f'P{pid}', 'x': round(x, 2), 'y': round(y, 2),
                                      'z': feet, 'model': 'walk_actor', 'support': 'water',
                                      'house': None, 'posture': 'standing', 'near_car': car['id'],
                                      'yaw': round(yaw, 3)})
        if walls:
            self.models.append(static_model('low_walls', 0, 0, 0, 0, walls))

    # -- optional fixed top-down camera for docs / eval ------------------------
    def overview_cam(self, alt: float = 75.0):
        self.models.append(
            '<model name="overview_cam"><static>true</static>'
            f'{pose(0, 0, alt, 0, math.pi / 2, math.pi / 2)}<link name="link">'
            '<sensor name="overview" type="camera"><camera><horizontal_fov>1.2</horizontal_fov>'
            '<image><width>1280</width><height>960</height><format>R8G8B8</format></image>'
            '<clip><near>1</near><far>200</far></clip></camera>'
            '<always_on>1</always_on><update_rate>0.5</update_rate>'
            '<topic>flood/overview/image</topic></sensor></link></model>')

    # -- assemble ---------------------------------------------------------------
    def build(self, overview: bool) -> str:
        self.ground()
        empty = self.houses()
        self.roads()
        empty = self.mounds(empty)
        self.water()
        cars = self.cars()
        self.trees_and_landmarks(empty)
        self.debris()
        self.people(cars)
        if overview:
            self.overview_cam()
        return world_sdf(self.cfg['world_name'], '\n    '.join(self.models))


def world_sdf(name: str, models: str) -> str:
    # Plugins and <spherical_coordinates> copied from drone_basic/worlds/home.sdf.
    return f"""<?xml version="1.0"?>
<!-- Generated by flood_vlm/scene_gen.py - do not edit by hand. -->
<sdf version="1.10">
  <world name="{name}">
    <physics name="4ms" type="ode">
      <max_step_size>0.01</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>
    <plugin name="gz::sim::systems::Physics" filename="gz-sim-physics-system"/>
    <plugin name="gz::sim::systems::UserCommands" filename="gz-sim-user-commands-system"/>
    <plugin name="gz::sim::systems::SceneBroadcaster" filename="gz-sim-scene-broadcaster-system"/>
    <plugin name="gz::sim::systems::Contact" filename="gz-sim-contact-system"/>
    <plugin name="gz::sim::systems::Sensors" filename="gz-sim-sensors-system">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin name="gz::sim::systems::Imu" filename="gz-sim-imu-system"/>
    <plugin name="gz::sim::systems::NavSat" filename="gz-sim-navsat-system"/>
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>47.478950</latitude_deg>
      <longitude_deg>19.057785</longitude_deg>
      <elevation>0</elevation>
      <heading_deg>0</heading_deg>
    </spherical_coordinates>
    <gravity>0 0 -9.8</gravity>
    <magnetic_field>5.5645e-06 2.28758e-05 -4.23884e-05</magnetic_field>
    <atmosphere type="adiabatic"/>
    <scene>
      <ambient>0.55 0.55 0.55 1</ambient>
      <background>0.62 0.66 0.70 1</background>
      <shadows>false</shadows>
      <grid>false</grid>
    </scene>
    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <visualize>false</visualize>
      <pose>0 0 50 0 0 0</pose>
      <diffuse>0.85 0.83 0.78 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <direction>-0.3 0.2 -0.9</direction>
    </light>
    {models}
  </world>
</sdf>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default=None, help='output .sdf (default: drone_basic/worlds/flood_town_<seed>.sdf)')
    ap.add_argument('--config', default=None, help='mission.yaml')
    ap.add_argument('--overview-cam', action='store_true', help='add a fixed top-down camera at 75 m')
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out = args.out or os.path.join(drone_worlds_dir(), f'flood_town_{args.seed}.sdf')
    gen = SceneGen(args.seed, cfg)
    sdf = gen.build(args.overview_cam)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w') as f:
        f.write(sdf)
    gt_path = os.path.splitext(out)[0] + '.gt.json'
    with open(gt_path, 'w') as f:
        json.dump(gen.gt, f, indent=1)
    n_models = sdf.count('<model ') + sdf.count('<include>') + sdf.count('<actor ')
    print(f'wrote {out} ({n_models} models, {len(gen.gt["people"])} people, '
          f'{len(gen.gt["houses"])} houses) and {gt_path}')


if __name__ == '__main__':
    main()
