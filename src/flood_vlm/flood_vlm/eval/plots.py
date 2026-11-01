#!/usr/bin/env python3
"""Ablation table + plots from results/vqa.csv and results/missions.csv (FLOOD_VLM_PLAN.md 4.4).

    python3 -m flood_vlm.eval.plots

Writes results/ablation.csv (one row per VQA configuration: overall accuracy,
latency split into prefill = vision encode + prompt, and decode),
results/pareto.png (accuracy vs latency) and results/latency_breakdown.png.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    args = ap.parse_args(argv)
    rows = read_csv(os.path.join(args.results, 'vqa.csv'))
    # keep the latest summary per (dataset, tag, type)
    latest = {}
    for r in rows:
        latest[(r['dataset'], r['tag'], r['type'])] = r
    configs = defaultdict(dict)
    for (ds, tag, typ), r in latest.items():
        configs[(ds, tag)][typ] = r
    abl = []
    for (ds, tag), types in sorted(configs.items()):
        a = types.get('ALL')
        if not a:
            continue
        abl.append({'dataset': ds, 'tag': tag, 'quant': a['quant'], 'long_side': a['long_side'], 'think': a['think'],
                    'n': a['n'], 'acc': a['acc'], 'wall_s': a['wall_s'],
                    'prefill_s': round(float(a['prompt_ms']) / 1000, 2), 'decode_s': round(float(a['gen_ms']) / 1000, 2),
                    'prompt_tok': a['prompt_tok'], 'gen_tok': a['gen_tok'],
                    **{f'acc_{t}': types[t]['acc'] for t in sorted(types) if t != 'ALL'}})
    if not abl:
        print('no VQA results yet')
        return
    keys = sorted({k for r in abl for k in r}, key=lambda k: list(abl[0]).index(k) if k in abl[0] else 99)
    with open(os.path.join(args.results, 'ablation.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(abl)
    print(f'{len(abl)} configs -> {args.results}/ablation.csv')

    # accuracy vs latency
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    marks = {'floodsim': 'o', 'floodnet': 's'}
    for r in abl:
        ax.scatter(float(r['wall_s']), float(r['acc']), marker=marks.get(r['dataset'], 'o'), s=60)
        ax.annotate(f"{r['dataset']}:{r['tag']}", (float(r['wall_s']), float(r['acc'])), fontsize=7,
                    xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel('seconds per question (CPU, llama.cpp)')
    ax.set_ylabel('accuracy')
    ax.set_title('Accuracy vs latency (Qwen3.5-4B)')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.results, 'pareto.png'), dpi=120)
    plt.close(fig)

    # latency breakdown
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [f"{r['dataset']}\n{r['tag']}" for r in abl]
    pre = [float(r['prefill_s']) for r in abl]
    dec = [float(r['decode_s']) for r in abl]
    ax.bar(labels, pre, label='vision encode + prompt prefill')
    ax.bar(labels, dec, bottom=pre, label='decode')
    ax.set_ylabel('seconds per question')
    ax.set_title('Where the time goes')
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.results, 'latency_breakdown.png'), dpi=120)
    plt.close(fig)
    print('plots -> pareto.png, latency_breakdown.png')


if __name__ == '__main__':
    main()
