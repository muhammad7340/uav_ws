"""Ground-truth rules over a scene's *.gt.json (FLOOD_VLM_PLAN.md 1.2).

Pure Python (no ROS) so it can be unit-tested and used offline by the scorer.
"""
from __future__ import annotations

import json
import math

LEVELS = ('critical', 'high', 'medium')
LEVEL_RANK = {lvl: i for i, lvl in enumerate(LEVELS)}

HIGH_MARGIN_M = 1.0          # roof person with roof_z - w below this is "high"
BEACON_DRY_MARGIN_M = 0.2    # beacon surface must be this far above the water
BEACON_MIN_M = 1.5
BEACON_MAX_M = 15.0


def load_gt(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def house_by_id(gt: dict, hid: str | None) -> dict | None:
    for h in gt['houses']:
        if h['id'] == hid:
            return h
    return None


def roof_margin(gt: dict, person: dict, w: float) -> float:
    """Height of the person's support above water (m). Negative = under water."""
    h = house_by_id(gt, person.get('house'))
    top = h['roof_z'] if h else person['z']
    return top - w


def urgency(gt: dict, person: dict, w: float) -> str:
    if person['support'] == 'water':
        return 'critical'
    if person['posture'] == 'lying':
        return 'critical'
    if person['support'] == 'roof' and roof_margin(gt, person, w) < HIGH_MARGIN_M:
        return 'high'
    return 'medium'


def ranking(gt: dict, w: float) -> list[dict]:
    """People sorted by (level, margin above water); each item carries level/margin."""
    out = []
    for p in gt['people']:
        out.append({'id': p['id'], 'level': urgency(gt, p, w),
                    'margin': roof_margin(gt, p, w)})
    out.sort(key=lambda r: (LEVEL_RANK[r['level']], r['margin']))
    return out


def point_in_polygon(x: float, y: float, poly: list[list[float]]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xc:
                inside = not inside
    return inside


def dry_zone_at(gt: dict, x: float, y: float) -> dict | None:
    """Highest dry zone containing (x, y), or None."""
    best = None
    for z in gt['dry_zones']:
        if point_in_polygon(x, y, z['polygon']):
            if best is None or z['top_z'] > best['top_z']:
                best = z
    return best


def valid_beacon(gt: dict, person: dict, x: float, y: float, w: float) -> tuple[bool, str]:
    """GT valid-spot rule: dry zone (or the person's own roof) above water, 1.5-15 m away."""
    d = math.hypot(x - person['x'], y - person['y'])
    if d < BEACON_MIN_M:
        return False, f'too close ({d:.1f} m)'
    if d > BEACON_MAX_M:
        return False, f'too far ({d:.1f} m)'
    zone = dry_zone_at(gt, x, y)
    if zone is None:
        return False, 'in water'
    if zone['type'] == 'roof' and zone.get('house') != person.get('house'):
        return False, f"on another roof ({zone.get('house')})"
    if zone['top_z'] <= w + BEACON_DRY_MARGIN_M:
        return False, f"zone {zone['id']} flooded (top {zone['top_z']:.1f} m)"
    return True, f"ok ({zone['id']}, {d:.1f} m)"


def surface_z(gt: dict, x: float, y: float, w: float) -> float:
    """Height of what is visible from above at (x, y): dry zone top or the water."""
    zone = dry_zone_at(gt, x, y)
    if zone is not None and zone['top_z'] > w:
        return zone['top_z']
    return w
