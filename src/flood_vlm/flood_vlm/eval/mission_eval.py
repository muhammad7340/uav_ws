#!/usr/bin/env python3
"""Headless closed-loop missions over several seeds (FLOOD_VLM_PLAN.md 4.2).

    source scripts/flood_env.sh; scripts/run_vlm.sh &
    ros2 run flood_vlm mission_eval --seeds 1..10 --lookagain on off
    ros2 run flood_vlm mission_eval --aggregate-only

Each mission is `flood_mission.launch.py headless:=true exit_on_done:=true` in
runs/eval/seed<S>_la<on|off>/; finished runs are skipped, so the job can be
resumed. All metrics.json files are collected into results/missions.csv plus a
mean row per look-again setting.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import time

FIELDS = ['seed', 'lookagain', 'plan_source', 'plan_valid_first_try', 'n_gt', 'n_detected', 'precision',
          'recall', 'loc_err_survey_m', 'loc_err_final_m', 'posture_acc', 'location_acc', 'kendall_tau',
          'critical_in_topk', 'level_acc', 'n_beacons', 'safe_beacon_rate', 'n_replans', 'replan_correct',
          'n_vlm_calls', 'n_json_errors', 'prompt_tokens', 'gen_tokens', 'vlm_s', 'flight_sim_s',
          'flight_wall_s', 'total_wall_s', 'ram_peak_mb']


def parse_seeds(spec: list[str]) -> list[int]:
    out = []
    for s in spec:
        if '..' in s:
            a, b = s.split('..')
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(s))
    return out


def run_one(seed: int, la: str, root: str, timeout_s: int) -> bool:
    run_id = f'seed{seed}_la{la}'
    rd = os.path.join(root, run_id)
    if os.path.exists(os.path.join(rd, 'metrics.json')) and 'done' in open(os.path.join(rd, 'events.jsonl')).read():
        print(f'{run_id}: done already')
        return True
    ws = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.realpath(__file__))))))
    world = os.path.join(ws, 'src', 'drone_basic', 'worlds', f'flood_town_{seed}.sdf')
    if not os.path.exists(world):
        subprocess.run(['python3', '-m', 'flood_vlm.scene_gen', '--seed', str(seed)], check=True)
        subprocess.run(['colcon', 'build', '--symlink-install', '--packages-select', 'drone_basic'],
                       cwd=ws, check=True, stdout=subprocess.DEVNULL)
    os.makedirs(rd, exist_ok=True)
    cmd = ['ros2', 'launch', 'flood_vlm', 'flood_mission.launch.py', f'seed:={seed}', 'headless:=true',
           f'lookagain:={"true" if la == "on" else "false"}', f'runs_root:={os.path.abspath(root)}',
           f'run_id:={run_id}', 'exit_on_done:=true']
    t0 = time.time()
    with open(os.path.join(rd, 'launch.log'), 'w') as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            print(f'{run_id}: timeout after {timeout_s} s')
    subprocess.run([os.path.join(ws, 'scripts', 'flood_stop.sh')], stdout=subprocess.DEVNULL)
    ok = os.path.exists(os.path.join(rd, 'metrics.json'))
    print(f'{run_id}: {"ok" if ok else "FAILED"} in {(time.time() - t0) / 60:.0f} min')
    return ok


def aggregate(root: str, out_csv: str):
    rows = []
    for path in sorted(glob.glob(os.path.join(root, 'seed*_la*', 'metrics.json'))):
        m = json.load(open(path))
        r = {k: m.get(k) for k in FIELDS}
        r['lookagain'] = 'on' if m.get('lookagain', True) else 'off'
        r['ram_peak_mb'] = (m.get('ram_peak_mb') or {}).get('total')
        rows.append(r)
    if not rows:
        print('no runs found')
        return
    for la in ('on', 'off'):
        rs = [r for r in rows if r['lookagain'] == la]
        if not rs:
            continue
        mean = {'seed': f'mean ({len(rs)})', 'lookagain': la}
        for k in FIELDS[2:]:
            vals = [r[k] for r in rs if isinstance(r[k], (int, float)) and not isinstance(r[k], bool)]
            bools = [r[k] for r in rs if isinstance(r[k], bool)]
            if vals:
                mean[k] = round(sum(vals) / len(vals), 3)
            elif bools:
                mean[k] = round(sum(bools) / len(bools), 3)
        rows.append(mean)
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print(f'{len(rows)} rows -> {out_csv}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', nargs='+', default=['1..10'])
    ap.add_argument('--lookagain', nargs='+', choices=['on', 'off'], default=['on', 'off'])
    ap.add_argument('--root', default='runs/eval')
    ap.add_argument('--out', default='results/missions.csv')
    ap.add_argument('--timeout-min', type=int, default=150)
    ap.add_argument('--aggregate-only', action='store_true')
    args = ap.parse_args(argv)
    if not args.aggregate_only:
        for seed in parse_seeds(args.seeds):
            for la in args.lookagain:
                run_one(seed, la, args.root, args.timeout_min * 60)
                aggregate(args.root, args.out)
    aggregate(args.root, args.out)


if __name__ == '__main__':
    main()
