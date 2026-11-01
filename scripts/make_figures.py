#!/usr/bin/env python3
"""Figures for docs/flood (FLOOD_VLM_PLAN.md Phase 5).

    python3 scripts/make_figures.py --run runs/seed7_full

* grounded_boxes.png: the survey frame with the most people, with the VLM's own boxes
  (people coloured by location, dry areas dashed) exactly as returned in vlm_calls.jsonl
* map_seed7.png: copy of the run's map.png
* latency_breakdown.png / pareto.png: copies from results/ when present
"""
import argparse
import json
import os
import shutil

import cv2

COL = {'on_roof': (0, 200, 0), 'in_water': (0, 0, 255), 'on_dry_ground': (255, 160, 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='runs/seed7_full')
    ap.add_argument('--out', default='docs/flood')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    calls = [json.loads(line) for line in open(os.path.join(args.run, 'vlm_calls.jsonl'))]
    surveys = [c for c in calls if c['prompt_id'].startswith('survey') and 'error' not in c['output']]
    best = max(surveys, key=lambda c: len(c['output'].get('people', [])))
    img = cv2.imread(best['image'])
    h, w = img.shape[:2]
    big = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    s = 2.0

    def px(b):
        return [int(b[0] / 1000 * w * s), int(b[1] / 1000 * h * s), int(b[2] / 1000 * w * s), int(b[3] / 1000 * h * s)]

    for d in best['output'].get('dry_areas', []):
        x1, y1, x2, y2 = px(d['bbox_2d'])
        for k in range(x1, x2, 14):
            cv2.line(big, (k, y1), (min(k + 7, x2), y1), (255, 255, 0), 2)
            cv2.line(big, (k, y2), (min(k + 7, x2), y2), (255, 255, 0), 2)
        for k in range(y1, y2, 14):
            cv2.line(big, (x1, k), (x1, min(k + 7, y2)), (255, 255, 0), 2)
            cv2.line(big, (x2, k), (x2, min(k + 7, y2)), (255, 255, 0), 2)
        cv2.putText(big, f"dry {d['kind']}", (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    for p in best['output'].get('people', []):
        x1, y1, x2, y2 = px(p['bbox_2d'])
        c = COL.get(p['location'], (255, 255, 255))
        cv2.rectangle(big, (x1, y1), (x2, y2), c, 3)
        cv2.putText(big, f"{p['posture']} {p['location']} {p['confidence']:.2f}", (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    tm = best['timings']      # compute time, not wall time (a suspend can sit inside wall_s)
    secs = ((tm.get('prompt_ms') or 0) + (tm.get('predicted_ms') or 0)) / 1000
    cv2.putText(big, f"Qwen3.5-4B survey ({best['prompt_id']}, {secs:.0f} s on CPU)",
                (10, 2 * h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    out = os.path.join(args.out, 'grounded_boxes.png')
    cv2.imwrite(out, big)
    print(out, len(best['output'].get('people', [])), 'people')

    shutil.copy(os.path.join(args.run, 'map.png'), os.path.join(args.out, 'map_seed7.png'))
    for f in ('latency_breakdown.png', 'pareto.png'):
        if os.path.exists(os.path.join('results', f)):
            shutil.copy(os.path.join('results', f), os.path.join(args.out, f))


if __name__ == '__main__':
    main()
