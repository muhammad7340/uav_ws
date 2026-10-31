#!/usr/bin/env python3
"""Phase 2 check: survey one cell at 20 m, ground people, match them to GT.

    source scripts/flood_env.sh
    ros2 launch flood_vlm flood_world.launch.py seed:=7 headless:=true water_schedule:=false &
    scripts/run_vlm.sh &
    python3 scripts/try_survey.py --seed 7 --look

Prints grounded people in world coordinates, Hungarian matching against the
people that are actually in the frame (3 m gate), precision/recall, and with
--look the localization error after a look-again at 8 m.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import rclpy

from flood_vlm import geometry as geo
from flood_vlm import gt as G
from flood_vlm import prompts, schemas, scoring
from flood_vlm.config import drone_worlds_dir, load_config
from flood_vlm.memory import ObjectMemory
from flood_vlm.perception import look_position, look_prompt, process_look, process_survey
from flood_vlm.ros_io import DroneIO
from flood_vlm.vlm_client import VLMClient


def gt_in_frame(gt, frame):
    """GT people whose projection lands inside the image."""
    w, h = frame.size
    out = []
    for p in gt['people']:
        uv = geo.world_to_pixel((p['x'], p['y'], p['z'] + 0.5), frame.K, frame.R, frame.t)
        if uv and 0 <= uv[0] < w and 0 <= uv[1] < h:
            out.append(p)
    return out


def best_cell(gt, cells):
    return max(cells, key=lambda c: sum(1 for p in gt['people']
                                        if abs(p['x'] - c[0]) < 12 and abs(p['y'] - c[1]) < 16))


def ask(vlm, frame, io, prompt, schema, max_tokens, pid):
    io.pause(True)
    try:
        return vlm.ask(frame.image, prompt, schema, think=False, max_tokens=max_tokens, prompt_id=pid)
    finally:
        io.pause(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--cell', type=float, nargs=2, default=None, help='world x y (default: busiest quadrant)')
    ap.add_argument('--alt', type=float, default=None)
    ap.add_argument('--look', action='store_true', help='look again at 8 m over each detection')
    ap.add_argument('--repeat', type=int, default=1, help='repeat the survey call on the same frame')
    ap.add_argument('--run-dir', default=None)
    args = ap.parse_args()

    cfg = load_config()
    gt = G.load_gt(os.path.join(drone_worlds_dir(), f'flood_town_{args.seed}.gt.json'))
    run_dir = args.run_dir or os.path.join('runs', time.strftime('try_survey_%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)
    vlm = VLMClient(cfg['llm']['url'], run_dir=run_dir, temperature=cfg['llm']['temperature'])
    if not vlm.healthy():
        raise SystemExit('llama-server is not answering: start scripts/run_vlm.sh')

    rclpy.init()
    io = DroneIO(cfg, 'try_survey')
    io.start()
    if not io.wait_ready():
        raise SystemExit('no odometry / camera_info: is flood_world.launch.py running?')
    alt = args.alt or cfg['survey_alt']
    cell = args.cell or best_cell(gt, [(-15, -15), (15, -15), (-15, 15), (15, 15)])
    print(f'cell {cell} at {alt} m; odom->world offset {np.round(io.ow.offset, 3).tolist()}')

    home = cfg['home']
    io.goto_world(home['x'], home['y'], alt)
    io.goto_world(cell[0], cell[1], alt)
    frame = io.grab()
    print(f'frame at sim {frame.stamp:.1f}s pose={frame.pose_source} cam={np.round(frame.t, 2).tolist()} '
          f'(gt drone {np.round(frame.drone_world, 2).tolist()})')

    mem = ObjectMemory(cfg['merge_radius_m'], run_dir)
    n_fail = 0
    timings = []
    out = {}
    for r in range(args.repeat):
        out, tm = ask(vlm, frame, io, prompts.P_SURVEY, schemas.SURVEY, 768, f'survey{r}')
        timings.append(tm)
        n_fail += 'error' in out
        print(f'survey call {r}: {tm["wall_s"]} s, prompt {tm.get("prompt_n")} tok, '
              f'gen {tm.get("predicted_n")} tok, finish={tm.get("finish_reason")}'
              + (f' ERROR {out["error"]}' if 'error' in out else ''))
    process_survey(out, frame, mem, 'survey') if 'error' not in out else []

    visible = gt_in_frame(gt, frame)
    det_xy = [p.xyz_world[:2] for p in mem.active()]
    sc = scoring.detection_scores(det_xy, visible)
    print(f'\nGT people in frame: {[p["id"] for p in visible]}')
    for p in mem.active():
        print(f'  {p.id}: world ({p.xyz_world[0]:6.2f}, {p.xyz_world[1]:6.2f}, {p.xyz_world[2]:4.1f}) '
              f'{p.location:13s} {p.posture:8s} conf {p.conf:.2f}')
    print(f'survey: precision {sc["precision"]:.2f} recall {sc["recall"]:.2f} '
          f'mean loc err {sc["loc_err_m"]} m (3 m gate)')
    survey_err = {}
    for i, j, d in sc['matches']:
        survey_err[mem.active()[i].id] = (visible[j]['id'], d)

    look_err = {}
    if args.look:
        for p in list(mem.active()):
            io.goto_world(*look_position(p, cfg['look_alt']))
            f2 = io.grab()
            o2, tm = ask(vlm, f2, io, look_prompt(p, f2), schemas.LOOK, cfg['llm']['max_tokens_fast'], f'look_{p.id}')
            timings.append(tm)
            n_fail += 'error' in o2
            res = process_look(o2, f2, p, mem, 'look') if 'error' not in o2 else {}
            g = min(gt['people'], key=lambda q: math.hypot(q['x'] - p.xyz_world[0], q['y'] - p.xyz_world[1]))
            d = math.hypot(g['x'] - p.xyz_world[0], g['y'] - p.xyz_world[1])
            look_err[p.id] = (g['id'], d)
            print(f'look {p.id}: {tm["wall_s"]} s present={res.get("person_present")} -> '
                  f'({p.xyz_world[0]:.2f}, {p.xyz_world[1]:.2f}) {p.posture}/{p.location}; '
                  f'nearest GT {g["id"]} at {d:.2f} m ({g["posture"]}/{g["support"]}); beacon {p.beacon_xy}')

    io.goto_world(home['x'], home['y'], alt)
    summary = {
        'seed': args.seed, 'cell': cell, 'alt': alt, 'gt_in_frame': [p['id'] for p in visible],
        'survey': {k: v for k, v in sc.items() if k != 'matches'},
        'survey_err': survey_err, 'look_err': look_err,
        'mean_err_survey_m': float(np.mean([v[1] for v in survey_err.values()])) if survey_err else None,
        'mean_err_look_m': float(np.mean([v[1] for v in look_err.values() if v[1] <= 3.0]))
        if look_err else None,
        'json_failures': n_fail, 'vlm_calls': len(timings),
        'vlm_wall_s': [t['wall_s'] for t in timings],
        'people': mem.to_dict()['people'],
    }
    with open(os.path.join(run_dir, 'try_survey.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'\nJSON failures: {n_fail}/{len(timings)}; summary in {run_dir}/try_survey.json')
    io.stop()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
