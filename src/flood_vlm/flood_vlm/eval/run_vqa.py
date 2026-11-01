#!/usr/bin/env python3
"""Run the VLM on FloodSim-VQA or FloodNet VQA (FLOOD_VLM_PLAN.md 4.2).

    ros2 run flood_vlm run_vqa --dataset floodsim --n 100 --tag q4_640
    ros2 run flood_vlm run_vqa --dataset floodnet --n 300 --tag q4_640
    ros2 run flood_vlm run_vqa --dataset floodsim --n 100 --long-side 448 --tag q4_448

Stratified sample per question type (fixed seed, so different --tag runs see the
same questions). Per-answer log: results/vqa/<dataset>_<tag>.jsonl (resumable);
summary rows (accuracy per type, counting also +-1, latency split) are appended to
results/vqa.csv. --quant only labels the run: start llama-server with QUANT=Q8_0.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from collections import defaultdict

import cv2

from flood_vlm import prompts, schemas
from flood_vlm.config import load_config
from flood_vlm.vlm_client import VLMClient

NUM_WORDS = {'zero': 0, 'none': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
             'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
SYNONYMS = {'not flooded': 'non flooded', 'non-flooded': 'non flooded', 'dry': 'non flooded',
            'unflooded': 'non flooded', 'medium': 'moderate', 'dense': 'high', 'sparse': 'low'}


def norm(a: str) -> str:
    a = str(a).strip().lower().rstrip('.').strip()
    a = re.sub(r'\s+', ' ', a)
    return SYNONYMS.get(a, a)


def to_int(a: str) -> int | None:
    a = norm(a)
    if a in NUM_WORDS:
        return NUM_WORDS[a]
    m = re.search(r'-?\d+', a)
    return int(m.group()) if m else None


def score(q: dict, pred) -> dict:
    """Returns {'correct': bool, 'pm1': bool|None}."""
    at = q['answer_type']
    if at == 'point':
        if not isinstance(pred, list) or len(pred) != 2:
            return {'correct': False}
        img = cv2.imread(q['image'])
        h, w = img.shape[:2]
        u, v = pred[0] / 1000 * w, pred[1] / 1000 * h
        x1, y1, x2, y2 = q['answer']
        m = 6     # px tolerance around the projected box
        return {'correct': x1 - m <= u <= x2 + m and y1 - m <= v <= y2 + m}
    if at == 'int':
        p = to_int(pred)
        gt = int(q['answer'])
        return {'correct': p == gt, 'pm1': p is not None and abs(p - gt) <= 1}
    p = norm(pred)
    g = norm(q['answer'])
    if at == 'yes_no':
        p = 'yes' if p.startswith('yes') else 'no' if p.startswith('no') else p
    if q.get('options') and p not in [norm(o) for o in q['options']]:
        # accept an answer that contains exactly one option word
        hits = [o for o in q['options'] if re.search(rf'\b{re.escape(norm(o))}\b', p)]
        p = norm(hits[0]) if len(hits) == 1 else p
    return {'correct': p == g}


def stratified(qs: list[dict], n: int, seed: int = 0) -> list[dict]:
    by = defaultdict(list)
    for q in qs:
        by[q['type']].append(q)
    rng = random.Random(seed)
    per = max(1, n // len(by))
    out = []
    for t in sorted(by):
        items = by[t][:]
        rng.shuffle(items)
        out += items[:per]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=['floodsim', 'floodnet'], required=True)
    ap.add_argument('--questions', default=None)
    ap.add_argument('--split', default='test')
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--think', choices=['on', 'off'], default='off')
    ap.add_argument('--quant', default='q4')
    ap.add_argument('--long-side', type=int, default=0, help='resize the image (0 = keep)')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--results', default='results')
    args = ap.parse_args(argv)

    cfg = load_config()
    qpath = args.questions or f'data/{args.dataset}/questions.jsonl'
    qs = [json.loads(line) for line in open(qpath)]
    qs = [q for q in qs if q.get('split', 'test') == args.split]
    sample = stratified(qs, args.n)
    tag = args.tag or f'{args.quant}_{args.long_side or "orig"}_think{args.think}'
    os.makedirs(os.path.join(args.results, 'vqa'), exist_ok=True)
    log_path = os.path.join(args.results, 'vqa', f'{args.dataset}_{tag}.jsonl')
    done = {}
    if os.path.exists(log_path):
        for line in open(log_path):
            r = json.loads(line)
            done[r['qid']] = r
    vlm = VLMClient(cfg['llm']['url'], temperature=0.0, image_long_side=args.long_side or None)
    if not vlm.healthy():
        raise SystemExit('llama-server is not answering (scripts/run_vlm.sh)')
    think = args.think == 'on'
    max_tokens = 1536 if think else 48
    t0 = time.time()
    with open(log_path, 'a') as log:
        for i, q in enumerate(sample):
            if q['qid'] in done:
                continue
            img = cv2.imread(q['image'])
            if q['answer_type'] == 'point':
                out, tm = vlm.ask(img, prompts.P_POINT.format(question=q['question']), schemas.POINT_ANSWER,
                                  think=think, max_tokens=max_tokens, prompt_id=q['qid'])
                pred = out.get('point_2d')
            else:
                out, tm = vlm.ask(img, prompts.P_VQA.format(question=q['question']), schemas.ANSWER,
                                  think=think, max_tokens=max_tokens, prompt_id=q['qid'])
                pred = out.get('answer', '')
            s = score(q, pred)
            r = {'qid': q['qid'], 'type': q['type'], 'question': q['question'], 'answer': q['answer'],
                 'pred': pred, **s, 'error': out.get('error'),
                 'prompt_n': tm.get('prompt_n'), 'prompt_ms': tm.get('prompt_ms'),
                 'gen_n': tm.get('predicted_n'), 'gen_ms': tm.get('predicted_ms'), 'wall_s': tm['wall_s']}
            done[q['qid']] = r
            log.write(json.dumps(r) + '\n')
            log.flush()
            el = time.time() - t0
            print(f'[{i + 1}/{len(sample)}] {q["type"]:16s} gt={str(q["answer"])[:12]:12s} '
                  f'pred={str(pred)[:16]:16s} {"OK " if s["correct"] else "-- "} {tm["wall_s"]:.1f}s '
                  f'(elapsed {el / 60:.0f} min)', flush=True)

    rows = [done[q['qid']] for q in sample if q['qid'] in done]
    summary = []
    by = defaultdict(list)
    for r in rows:
        by[r['type']].append(r)
    by['ALL'] = rows
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    for t, rs in sorted(by.items()):
        pm1 = [r['pm1'] for r in rs if r.get('pm1') is not None]
        summary.append({
            'dataset': args.dataset, 'tag': tag, 'quant': args.quant, 'long_side': args.long_side or 'orig',
            'think': args.think, 'type': t, 'n': len(rs),
            'acc': round(mean([r['correct'] for r in rs]), 3),
            'acc_pm1': round(mean(pm1), 3) if pm1 else '',
            'json_err': sum(1 for r in rs if r.get('error')),
            'wall_s': round(mean([r['wall_s'] for r in rs]), 2),
            'prompt_ms': round(mean([r['prompt_ms'] or 0 for r in rs])),
            'gen_ms': round(mean([r['gen_ms'] or 0 for r in rs])),
            'prompt_tok': round(mean([r['prompt_n'] or 0 for r in rs])),
            'gen_tok': round(mean([r['gen_n'] or 0 for r in rs])),
        })
    csv_path = os.path.join(args.results, 'vqa.csv')
    new = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        if new:
            wr.writeheader()
        wr.writerows(summary)
    for s in summary:
        print(f"{s['type']:18s} n={s['n']:4d} acc={s['acc']:.3f} {('pm1=' + str(s['acc_pm1'])) if s['acc_pm1'] != '' else ''}"
              f"  {s['wall_s']} s/q")


if __name__ == '__main__':
    main()
