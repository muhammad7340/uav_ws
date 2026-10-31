"""Scoring against scene ground truth (FLOOD_VLM_PLAN.md 4.3)."""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from flood_vlm import gt as G

GATE_M = 3.0
LOCATION_OF_SUPPORT = {'roof': 'on_roof', 'water': 'in_water', 'mound': 'on_dry_ground'}


def match(det_xy: list, gt_xy: list, gate: float = GATE_M) -> list[tuple[int, int, float]]:
    """Hungarian matching with a distance gate. Returns (det_idx, gt_idx, dist)."""
    if not det_xy or not gt_xy:
        return []
    d = np.array([[math.hypot(a[0] - b[0], a[1] - b[1]) for b in gt_xy] for a in det_xy])
    cost = np.where(d <= gate, d, 1e6)
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c), float(d[r, c])) for r, c in zip(rows, cols) if d[r, c] <= gate]


def detection_scores(det_xy: list, gt_people: list[dict], gate: float = GATE_M) -> dict:
    gt_xy = [(p['x'], p['y']) for p in gt_people]
    m = match(det_xy, gt_xy, gate)
    tp = len(m)
    return {
        'tp': tp, 'n_det': len(det_xy), 'n_gt': len(gt_xy),
        'precision': tp / len(det_xy) if det_xy else 0.0,
        'recall': tp / len(gt_xy) if gt_xy else 0.0,
        'loc_err_m': float(np.mean([x[2] for x in m])) if m else None,
        'matches': m,
    }


def kendall_tau(order_a: list[str], order_b: list[str]) -> float | None:
    """Kendall's tau between two rankings over their common items."""
    common = [x for x in order_a if x in order_b]
    n = len(common)
    if n < 2:
        return None
    pos_b = {x: i for i, x in enumerate(order_b)}
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = pos_b[common[i]] - pos_b[common[j]]
            if s < 0:
                conc += 1
            elif s > 0:
                disc += 1
    return (conc - disc) / (n * (n - 1) / 2)


def score_mission(gt: dict, memory_people: list[dict], ranking: list[dict], beacons: list[dict],
                  replans: list[dict], water_final: float) -> dict:
    """Mission metrics for metrics.json. memory_people are Person dicts (asdict)."""
    people = [p for p in memory_people if p.get('status') != 'rejected']
    det_xy = [p['xyz_world'][:2] for p in people]
    ds = detection_scores(det_xy, gt['people'])
    by_det = {people[i]['id']: gt['people'][j] for i, j, _ in ds['matches']}

    survey_err, final_err, post_ok, loc_ok = [], [], [], []
    for p in people:
        g = by_det.get(p['id'])
        if g is None:
            continue
        final_err.append(math.hypot(p['xyz_world'][0] - g['x'], p['xyz_world'][1] - g['y']))
        if p.get('survey_xyz'):
            survey_err.append(math.hypot(p['survey_xyz'][0] - g['x'], p['survey_xyz'][1] - g['y']))
        post_ok.append(p['posture'] == g['posture'])
        loc_ok.append(p['location'] == LOCATION_OF_SUPPORT.get(g['support']))

    # triage vs GT ranking at the water level the final triage saw
    gt_rank = [r['id'] for r in G.ranking(gt, water_final)]
    gt_level = {r['id']: r['level'] for r in G.ranking(gt, water_final)}
    det_order_as_gt = [by_det[r['person_id']]['id'] for r in ranking if r['person_id'] in by_det]
    tau = kendall_tau(det_order_as_gt, gt_rank)
    n_crit = sum(1 for v in gt_level.values() if v == 'critical')
    top = det_order_as_gt[:n_crit]
    crit_topk = (sum(1 for x in top if gt_level[x] == 'critical') / n_crit) if n_crit else None
    level_ok = [gt_level[by_det[r['person_id']]['id']] == r['level'] for r in ranking
                if r['person_id'] in by_det]

    # beacons: GT valid-spot rule at the water level when placed
    safe = []
    for b in beacons:
        g = by_det.get(b['person_id'])
        if g is None:
            safe.append(False)
            continue
        ok, _ = G.valid_beacon(gt, g, b['x'], b['y'], b['water'])
        safe.append(ok)

    # re-plan: after a rise, did people whose GT level got worse move up?
    # re-plan: each person who became more urgent in GT must have been moved up the
    # remaining order, given a more urgent level, or already be next in line
    replan_ok = []
    for ev in replans:
        worse = ev.get('gt_newly_urgent', [])
        if not worse:
            continue
        rest = [x for x in ev['new_order'] if x not in ev.get('done', [])]
        ok = True
        for pid in worse:
            lv_old = G.LEVEL_RANK.get(ev.get('old_levels', {}).get(pid), 9)
            lv_new = G.LEVEL_RANK.get(ev.get('new_levels', {}).get(pid), 9)
            ok &= pid in ev.get('moved_up', []) or lv_new < lv_old or (rest and rest[0] == pid)
        replan_ok.append(bool(ok))

    mean = (lambda xs: float(np.mean(xs)) if xs else None)
    return {
        'n_gt': ds['n_gt'], 'n_detected': ds['n_det'], 'tp': ds['tp'],
        'precision': ds['precision'], 'recall': ds['recall'],
        'loc_err_survey_m': mean(survey_err), 'loc_err_final_m': mean(final_err),
        'posture_acc': mean(post_ok), 'location_acc': mean(loc_ok),
        'kendall_tau': tau, 'critical_in_topk': crit_topk, 'level_acc': mean(level_ok),
        'n_beacons': len(beacons), 'safe_beacon_rate': mean(safe),
        'replan_events': len(replans), 'replan_correct': mean(replan_ok),
        'gt_ranking': gt_rank, 'det_ranking_as_gt': det_order_as_gt,
    }
