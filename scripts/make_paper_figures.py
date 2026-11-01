#!/usr/bin/env python3
"""All FloodScout-VLM figures and tables, reproducible from the repo data.

    python3 scripts/make_paper_figures.py               # everything into assets/floodscout/

Inputs: results/vqa.csv + results/vqa/*.jsonl, data/floodsim, data/floodnet, runs/<id>/ (mission
runs: metrics.json, memory.json, vlm_calls.jsonl, events.jsonl, rec/ frames from
scripts/record_showcase.sh), runs/try_survey_*/try_survey.json, the scene GT json files.
Images carry (almost) no text so they scale; captions live in the README.
Tables are written as Markdown to assets/floodscout/tables/.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

WS = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(WS, 'src', 'flood_vlm'))
from flood_vlm import geometry as geo  # noqa: E402
from flood_vlm import gt as G  # noqa: E402

OUT = os.path.join(WS, 'assets', 'floodscout')
TAB = os.path.join(OUT, 'tables')
GRD = os.path.join(OUT, 'grounding')

# palette (colour-blind safe, Okabe-Ito)
C = {'sim640': '#0072B2', 'sim448': '#56B4E9', 'fn': '#E69F00', 'fn_primed': '#999999',
     'prefill': '#0072B2', 'decode': '#E69F00', 'flight': '#009E73', 'survey': '#0072B2',
     'look': '#56B4E9', 'plan': '#CC79A7', 'triage': '#D55E00', 'v1': '#999999', 'final': '#0072B2',
     'show': '#009E73', 'critical': '#D55E00', 'high': '#E69F00', 'medium': '#0072B2'}
TYPES = ['simple_counting', 'complex_counting', 'condition', 'yes_no', 'grounding', 'priority', 'density']
TLAB = {'simple_counting': 'count', 'complex_counting': 'complex\ncount', 'condition': 'condition',
        'yes_no': 'yes/no', 'grounding': 'grounding', 'priority': 'priority', 'density': 'density',
        'ALL': 'all'}
BGR = {'on_roof': (60, 200, 60), 'in_water': (40, 40, 230), 'on_dry_ground': (230, 160, 40)}

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 200, 'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.25,
                     'grid.linestyle': '-', 'axes.axisbelow': True})


def p(*a):
    return os.path.join(WS, *a)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print('  ', name)


def jl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def vqa_latest():
    rows = {}
    for r in csv.DictReader(open(p('results', 'vqa.csv'))):
        rows[(r['dataset'], r['tag'], r['type'])] = r
    return rows


def md_table(path, header, rows):
    with open(path, 'w') as f:
        f.write('| ' + ' | '.join(header) + ' |\n|' + '---|' * len(header) + '\n')
        for r in rows:
            f.write('| ' + ' | '.join(str(v) for v in r) + ' |\n')
    print('  ', os.path.relpath(path, OUT))


def fmt(v, d=2):
    return '-' if v is None or v == '' else (f'{float(v):.{d}f}' if isinstance(v, (int, float, str)) else str(v))


def mosaic(images, cols, size=(640, 480), gap=6):
    """Grid of BGR images on white, all resized to `size` (no text)."""
    w, h = size
    rows = math.ceil(len(images) / cols)
    out = np.full((rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap, 3), 255, np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        out[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    return out


def crop_center(im, aspect):
    h, w = im.shape[:2]
    if w / h > aspect:
        nw = int(h * aspect)
        x0 = (w - nw) // 2
        return im[:, x0:x0 + nw]
    nh = int(w / aspect)
    y0 = (h - nh) // 2
    return im[y0:y0 + nh]


# ───────── grounding overlays (VLM output on the frame it saw, no text) ──────
def draw_boxes(img, out, gt_boxes=()):
    im = img.copy()
    h, w = im.shape[:2]
    s = max(1, round(w / 640))

    def px(b):
        return [int(b[0] / 1000 * w), int(b[1] / 1000 * h), int(b[2] / 1000 * w), int(b[3] / 1000 * h)]
    for d in out.get('dry_areas', []) or []:
        x1, y1, x2, y2 = px(d['bbox_2d'])
        ov = im.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), (255, 230, 0), -1)
        im = cv2.addWeighted(ov, 0.12, im, 0.88, 0)
        cv2.rectangle(im, (x1, y1), (x2, y2), (255, 230, 0), s)
    for b in gt_boxes:
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(im, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), 1)
    people = out.get('people')
    if people is None and out.get('person_present') and out.get('bbox_2d'):
        people = [out]
    for d in people or []:
        x1, y1, x2, y2 = px(d['bbox_2d'])
        cv2.rectangle(im, (x1, y1), (x2, y2), BGR.get(d.get('location'), (255, 255, 255)), 2 * s)
    bp = out.get('beacon_point_2d')
    if bp and out.get('person_present'):
        c = (int(bp[0] / 1000 * w), int(bp[1] / 1000 * h))
        cv2.circle(im, c, 7 * s, (0, 140, 255), -1)
        cv2.circle(im, c, 7 * s, (0, 0, 0), 1)
    return im


def fig_grounding(runs):
    os.makedirs(GRD, exist_ok=True)
    best = []
    for run in runs:
        calls = jl(p('runs', run, 'vlm_calls.jsonl'))
        for c in calls:
            if not c.get('image') or 'error' in c['output'] or not os.path.exists(c['image']):
                continue
            pid = c['prompt_id']
            if not (pid.startswith('survey') or pid.startswith('look')):
                continue
            im = draw_boxes(cv2.imread(c['image']), c['output'])
            cv2.imwrite(os.path.join(GRD, f'{run}_{c["i"]:03d}_{pid}.jpg'), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            n = len(c['output'].get('people', [])) if pid.startswith('survey') else int(bool(c['output'].get('person_present')))
            best.append((pid.startswith('survey'), n, im))
    print(f'   grounding/ ({len(best)} frames)')
    surveys = sorted([b for b in best if b[0]], key=lambda b: -b[1])[:3]
    looks = sorted([b for b in best if not b[0]], key=lambda b: -b[1])[:3]
    sel = [b[2] for b in surveys + looks]
    if sel:
        cv2.imwrite(os.path.join(OUT, 'fig_grounding_collage.jpg'), mosaic(sel, 3), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print('   fig_grounding_collage.jpg')


# ───────── environment ───────────────────────────────────────────────────────
def fig_environment(show_run):
    rec = p('runs', show_run, 'rec')
    tiles = []
    sc = sorted(glob.glob(os.path.join(rec, 'showcase', '*.jpg')))
    if sc:
        tiles.append(cv2.imread(sc[min(len(sc) - 1, 8)]))
        tiles.append(cv2.imread(sc[-1]))
    if os.path.exists(p('docs', 'flood', 'scene_seed7.png')):
        tiles.append(crop_center(cv2.imread(p('docs', 'flood', 'scene_seed7.png')), 16 / 9))
    dc = sorted(glob.glob(os.path.join(rec, 'dronecam', '*.jpg')))
    if dc:
        d = cv2.imread(dc[min(2, len(dc) - 1)])      # on the pad, before take-off
        h, w = d.shape[:2]
        tiles.append(d[int(h * 0.42):int(h * 0.98), int(w * 0.25):int(w * 0.75)])
    if len(tiles) == 4:
        cv2.imwrite(os.path.join(OUT, 'fig_environment.jpg'), mosaic(tiles, 2, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print('   fig_environment.jpg')
        cv2.imwrite(os.path.join(OUT, 'uav.jpg'), tiles[3], [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(OUT, 'sim_oblique.jpg'), tiles[0], [cv2.IMWRITE_JPEG_QUALITY, 92])
    down = sorted(glob.glob(os.path.join(rec, 'down', '*.jpg')))
    return down


# ───────── dataset ───────────────────────────────────────────────────────────
def fig_dataset():
    from flood_vlm.eval.floodsim_vqa_gen import Cam, visible_people
    qs = jl(p('data', 'floodsim', 'questions.jsonl'))
    frames = {f['image']: f for f in jl(p('data', 'floodsim', 'frames.jsonl'))}
    fn = jl(p('data', 'floodnet', 'questions.jsonl'))
    # collage: 4 FloodSim (20/12 m x water 0.8/2.4) with GT boxes + 4 FloodNet
    rng = random.Random(3)
    tiles = []
    for alt, w in ((20.0, 0.8), (20.0, 2.4), (12.0, 0.8), (12.0, 2.4)):
        cands = [f for f in frames.values() if f['alt'] == alt and f['water'] == w]
        gts = {}
        best = None
        for f in rng.sample(cands, min(40, len(cands))):
            g = gts.setdefault(f['seed'], G.load_gt(p('src', 'drone_basic', 'worlds', f'flood_town_{f["seed"]}.gt.json')))
            vis = visible_people(Cam(f), g, w)
            if best is None or len(vis) > len(best[1]):
                best = (f, vis)
        f, vis = best
        im = cv2.imread(f['image'])
        for v in vis:
            x1, y1, x2, y2 = [int(c) for c in v['box']]
            cv2.rectangle(im, (x1, y1), (x2, y2), BGR['in_water'] if v['support'] == 'water' else BGR['on_roof'], 2)
        tiles.append(im)
    imgs = sorted({q['image'] for q in fn})
    for f in random.Random(5).sample(imgs, 4):
        tiles.append(cv2.imread(f))
    cv2.imwrite(os.path.join(OUT, 'fig_dataset.jpg'), mosaic(tiles, 4, (480, 360)), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print('   fig_dataset.jpg')

    # stats figure
    types_fs = [t for t in TYPES if t != 'density']
    tr = Counter(q['type'] for q in qs if q['split'] == 'train')
    te = Counter(q['type'] for q in qs if q['split'] == 'test')
    fnc = Counter(q['type'] for q in fn)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={'width_ratios': [6, 5]})
    x = np.arange(len(types_fs))
    axes[0].bar(x, [tr[t] for t in types_fs], color=C['sim640'], label='train (seeds 1-8)')
    axes[0].bar(x, [te[t] for t in types_fs], bottom=[tr[t] for t in types_fs], color=C['sim448'], label='test (seeds 9-10)')
    axes[0].set_xticks(x, [TLAB[t] for t in types_fs])
    axes[0].set_ylabel('questions')
    axes[0].legend(frameon=False, fontsize=9)
    t2 = ['simple_counting', 'complex_counting', 'condition', 'yes_no', 'density']
    axes[1].bar(np.arange(len(t2)), [fnc[t] for t in t2], color=C['fn'])
    axes[1].set_xticks(np.arange(len(t2)), [TLAB[t] for t in t2])
    fig.tight_layout()
    save(fig, 'fig_dataset_stats.png')

    # table
    ex = {}
    for q in qs:
        ex.setdefault(q['type'], q)
    ans = defaultdict(Counter)
    for q in qs:
        a = q['answer'] if q['answer_type'] != 'point' else 'box'
        ans[q['type']][str(a)] += 1
    rows = []
    for t in types_fs:
        top = ', '.join(f'{a} ({n})' for a, n in ans[t].most_common(3))
        rows.append([TLAB[t].replace('\n', ' '), tr[t], te[t], tr[t] + te[t], ex[t]['question'].replace('|', '/'), top])
    rows.append(['**total**', sum(tr.values()), sum(te.values()), len(qs), '', ''])
    md_table(os.path.join(TAB, 'dataset_floodsim.md'),
             ['type', 'train', 'test', 'total', 'example question', 'most common answers'], rows)
    nf = len({f['image'] for f in frames.values()})
    md_table(os.path.join(TAB, 'dataset_overview.md'),
             ['dataset', 'images', 'questions', 'image size', 'source', 'answers'],
             [['FloodSim-VQA (ours)', nf, len(qs), '640 x 480', 'Gazebo, 10 seeds, 20/12 m, water 0.8/2.4 m',
               'exact, from scene GT'],
              ['FloodNet VQA (test)', len(imgs), len(fn), '224 x 224 (mirror)', 'real UAV, Hurricane Harvey',
               'human-labelled']])


# ───────── VQA results ───────────────────────────────────────────────────────
def fig_vqa():
    R = vqa_latest()
    cfg = [('floodsim', 'q4_640', 'sim640', 'FloodSim 640'), ('floodsim', 'q4_448', 'sim448', 'FloodSim 448'),
           ('floodnet', 'q4_224', 'fn', 'FloodNet')]
    types = [t for t in TYPES] + ['ALL']
    fig, ax = plt.subplots(figsize=(11, 3.8))
    wbar = 0.26
    for k, (ds, tag, col, lab) in enumerate(cfg):
        xs, ys = [], []
        for i, t in enumerate(types):
            r = R.get((ds, tag, t))
            if r:
                xs.append(i + (k - 1) * wbar)
                ys.append(float(r['acc']))
        ax.bar(xs, ys, wbar, color=C[col], label=lab)
    ax.set_xticks(range(len(types)), [TLAB[t] for t in types])
    ax.set_ylim(0, 1)
    ax.set_ylabel('accuracy')
    ax.legend(frameon=False, ncol=3, loc='upper left', fontsize=11)
    save(fig, 'fig_vqa_by_type.png')

    # prompt bias
    t5 = ['simple_counting', 'complex_counting', 'condition', 'yes_no', 'density', 'ALL']
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for k, (tag, col, lab) in enumerate((('q4_224_primed', 'fn_primed', '"flooded area" prompt'),
                                          ('q4_224', 'fn', 'neutral prompt'))):
        ax.bar(np.arange(len(t5)) + (k - 0.5) * 0.38, [float(R[('floodnet', tag, t)]['acc']) for t in t5], 0.38,
               color=C[col], label=lab)
    ax.set_xticks(range(len(t5)), [TLAB[t] for t in t5])
    ax.set_ylim(0, 1)
    ax.set_ylabel('accuracy')
    ax.legend(frameon=False, fontsize=9)
    save(fig, 'fig_prompt_bias.png')

    # latency breakdown
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    conf = [('floodnet', 'q4_224', '224 px'), ('floodsim', 'q4_448', '448 px'), ('floodsim', 'q4_640', '640 px')]
    pre = [float(R[(d, t, 'ALL')]['prompt_ms']) / 1000 for d, t, _ in conf]
    dec = [float(R[(d, t, 'ALL')]['gen_ms']) / 1000 for d, t, _ in conf]
    tok = [float(R[(d, t, 'ALL')]['prompt_tok']) for d, t, _ in conf]
    x = np.arange(len(conf))
    ax.bar(x, pre, 0.55, color=C['prefill'], label='vision encode + prefill')
    ax.bar(x, dec, 0.55, bottom=pre, color=C['decode'], label='decode')
    for i in x:
        ax.text(i, pre[i] + dec[i] + 0.4, f'{int(tok[i])} tok', ha='center', fontsize=9)
    ax.set_xticks(x, [c[2] for c in conf])
    ax.set_ylabel('seconds / question')
    ax.legend(frameon=False, fontsize=9)
    save(fig, 'fig_latency_breakdown.png')

    # accuracy vs latency
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for d, t, col, lab in (('floodsim', 'q4_640', 'sim640', 'FloodSim 640'), ('floodsim', 'q4_448', 'sim448', 'FloodSim 448'),
                           ('floodnet', 'q4_224', 'fn', 'FloodNet 224'), ('floodnet', 'q4_224_primed', 'fn_primed', 'FloodNet, biased prompt')):
        r = R[(d, t, 'ALL')]
        ax.scatter(float(r['wall_s']), float(r['acc']), s=90, color=C[col], label=lab, edgecolor='k', linewidth=0.5)
    ax.set_xlabel('seconds / question (CPU)')
    ax.set_ylabel('accuracy')
    ax.set_xlim(0, 25)
    ax.set_ylim(0.3, 0.65)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    save(fig, 'fig_pareto.png')

    # counting error + condition confusion
    def per_q(ds, tag):
        path = p('results', 'vqa', f'{ds}_{tag}.jsonl')
        return jl(path) if os.path.exists(path) else []
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    for ax, (ds, tag, col) in zip(axes, (('floodsim', 'q4_640', 'sim640'), ('floodnet', 'q4_224', 'fn'))):
        err = []
        for r in per_q(ds, tag):
            if r['type'] in ('simple_counting', 'complex_counting'):
                m = __import__('re').search(r'-?\d+', str(r['pred']))
                if m:
                    err.append(max(-6, min(6, int(m.group()) - int(r['answer']))))
        ax.hist(err, bins=np.arange(-6.5, 7.5, 1), color=C[col], edgecolor='white')
        ax.axvline(0, color='k', lw=0.8)
        ax.set_xlabel('predicted - true count')
    axes[0].set_ylabel('questions')
    save(fig, 'fig_count_error.png')

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    # left: FloodSim road condition; right: FloodNet yes/no (shows the model's "yes" bias)
    for ax, (ds, tag, typ, labels) in zip(axes, (('floodsim', 'q4_640', 'condition', ['passable', 'flooded']),
                                                 ('floodnet', 'q4_224', 'yes_no', ['yes', 'no']))):
        rs = [r for r in per_q(ds, tag) if r['type'] == typ and r['answer'] in labels]
        m = np.zeros((2, 2), int)
        for r in rs:
            pr = str(r['pred']).strip().lower().replace('not flooded', 'non flooded')
            if pr in labels:
                m[labels.index(r['answer']), labels.index(pr)] += 1
        ax.imshow(m, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, m[i, j], ha='center', va='center', fontsize=12,
                        color='white' if m[i, j] > m.max() / 2 else 'black')
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels)
        ax.set_xlabel('predicted')
        ax.grid(False)
    axes[0].set_ylabel('true')
    save(fig, 'fig_condition_confusion.png')

    # tables
    rows = []
    for ds, tag, lab in (('floodsim', 'q4_640', 'FloodSim-VQA test, 640 x 480'), ('floodsim', 'q4_448', 'FloodSim-VQA test, 448 px'),
                         ('floodnet', 'q4_224', 'FloodNet VQA test, zero-shot'),
                         ('floodnet', 'q4_224_primed', 'FloodNet, "flooded area" prompt')):
        row = [lab, R[(ds, tag, 'ALL')]['n']]
        for t in ('ALL',) + tuple(TYPES):
            r = R.get((ds, tag, t))
            row.append(fmt(r['acc']) if r else '-')
        row.append(fmt(R[(ds, tag, 'ALL')]['acc_pm1']))
        row.append(fmt(R[(ds, tag, 'ALL')]['wall_s'], 1))
        rows.append(row)
    md_table(os.path.join(TAB, 'vqa.md'),
             ['set', 'n', '**all**', 'count', 'complex count', 'condition', 'yes/no', 'grounding', 'priority', 'density',
              'counts +-1', 's / q'], rows)
    lat = []
    for d, t, lab in conf:
        r = R[(d, t, 'ALL')]
        tot = float(r['prompt_ms']) + float(r['gen_ms'])
        lat.append([lab, int(float(r['prompt_tok'])), fmt(float(r['prompt_ms']) / 1000, 1), fmt(float(r['gen_ms']) / 1000, 1),
                    f'{100 * float(r["prompt_ms"]) / tot:.0f} %', fmt(r['wall_s'], 1), fmt(r['acc'])])
    md_table(os.path.join(TAB, 'latency.md'),
             ['image', 'prompt tokens', 'prefill s', 'decode s', 'prefill share', 'total s', 'accuracy'], lat)


# ───────── missions ──────────────────────────────────────────────────────────
def mission_runs():
    out = []
    for run, lab, col in (('seed7_v1', 'run 1 (before fixes)', 'v1'), ('seed7_full', 'run 2', 'final'),
                          ('showcase_seed7', 'run 3 (recorded)', 'show')):
        mp = p('runs', run, 'metrics.json')
        if os.path.exists(mp):
            out.append((run, lab, col, json.load(open(mp))))
    return out


def fig_missions():
    runs = mission_runs()
    keys = [('recall', 'recall'), ('precision', 'precision'), ('location_acc', 'roof / water'),
            ('posture_acc', 'posture'), ('level_acc', 'urgency level'), ('critical_in_topk', 'critical\nin top-k'),
            ('safe_beacon_rate', 'safe beacon'), ('replan_correct', 're-plan')]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    wb = 0.8 / max(1, len(runs))
    for k, (_, lab, col, m) in enumerate(runs):
        ax.bar(np.arange(len(keys)) + (k - (len(runs) - 1) / 2) * wb,
               [m.get(key) or 0 for key, _ in keys], wb, color=C[col], label=lab)
    ax.set_xticks(range(len(keys)), [l for _, l in keys])
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc='upper right')
    save(fig, 'fig_mission_metrics.png')

    # localisation error: survey vs look-again, per matched person (all runs + try_survey)
    surv, look = [], []
    for run, _, _, m in runs:
        mem = json.load(open(p('runs', run, 'memory.json')))
        gt = G.load_gt(p('src', 'drone_basic', 'worlds', 'flood_town_7.gt.json'))
        for q in mem['people']:
            if q['status'] == 'rejected':
                continue
            g = min(gt['people'], key=lambda g: math.hypot(g['x'] - q['xyz_world'][0], g['y'] - q['xyz_world'][1]))
            d = math.hypot(g['x'] - q['xyz_world'][0], g['y'] - q['xyz_world'][1])
            if d > 3:
                continue
            if q.get('survey_xyz'):
                surv.append(math.hypot(g['x'] - q['survey_xyz'][0], g['y'] - q['survey_xyz'][1]))
            if q.get('looked'):
                look.append(d)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    bp = ax.boxplot([surv, look], widths=0.5, patch_artist=True, showfliers=False)
    for patch, col in zip(bp['boxes'], (C['survey'], C['look'])):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    for i, vals in enumerate((surv, look)):
        ax.scatter(np.random.default_rng(i).normal(i + 1, 0.05, len(vals)), vals, s=14, color='k', zorder=3)
    ax.set_xticks([1, 2], ['20 m survey', 'look-again\n+ triangulation'])
    ax.set_ylabel('position error (m)')
    save(fig, 'fig_localization.png')

    # urgency level confusion (mission triage vs GT at the triage water level)
    levels = ['critical', 'high', 'medium']
    M = np.zeros((3, 3), int)
    for run, _, _, m in runs:
        mem = json.load(open(p('runs', run, 'memory.json')))
        gt = G.load_gt(p('src', 'drone_basic', 'worlds', 'flood_town_7.gt.json'))
        w = m.get('water_final', 2.4)
        gl = {r['id']: r['level'] for r in G.ranking(gt, w)}
        for r in mem['ranking']:
            q = next(x for x in mem['people'] if x['id'] == r['person_id'])
            g = min(gt['people'], key=lambda g: math.hypot(g['x'] - q['xyz_world'][0], g['y'] - q['xyz_world'][1]))
            if math.hypot(g['x'] - q['xyz_world'][0], g['y'] - q['xyz_world'][1]) <= 3:
                M[levels.index(gl[g['id']]), levels.index(r['level'])] += 1
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(M, cmap='Oranges')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, M[i, j], ha='center', va='center', color='white' if M[i, j] > M.max() / 2 else 'black')
    ax.set_xticks(range(3), levels)
    ax.set_yticks(range(3), levels)
    ax.set_xlabel('predicted')
    ax.set_ylabel('ground truth')
    ax.grid(False)
    save(fig, 'fig_urgency_confusion.png')

    # VLM compute per call type (prefill / decode)
    per = defaultdict(lambda: [[], []])
    for run, _, _, _ in runs:
        for c in jl(p('runs', run, 'vlm_calls.jsonl')):
            k = c['prompt_id'].rstrip('0123456789').split('_')[0]
            tm = c['timings']
            if tm.get('prompt_ms') is None:
                continue
            per[k][0].append(tm['prompt_ms'] / 1000)
            per[k][1].append(tm['predicted_ms'] / 1000)
    order = [k for k in ('plan', 'survey', 'look', 'triage') if k in per]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    pre = [np.mean(per[k][0]) for k in order]
    dec = [np.mean(per[k][1]) for k in order]
    ax.bar(order, pre, 0.6, color=C['prefill'], label='vision encode + prefill')
    ax.bar(order, dec, 0.6, bottom=pre, color=C['decode'], label='decode')
    ax.set_ylabel('seconds / call')
    ax.legend(frameon=False, fontsize=9)
    save(fig, 'fig_vlm_calls.png')

    # mission timeline: flight vs VLM compute, in order (compute time, so a suspend does not show)
    fig, ax = plt.subplots(figsize=(11, 1.9 + 0.5 * len(runs)))
    for row, (run, lab, _, _) in enumerate(runs):
        ev = jl(p('runs', run, 'events.jsonl'))
        calls = jl(p('runs', run, 'vlm_calls.jsonl'))
        t, last_sim, ci = 0.0, 0.0, 0
        for e in ev:
            if e['kind'] != 'vlm' or ci >= len(calls):
                continue
            fl = max(0.0, e['t_sim'] - last_sim)
            ax.barh(row, fl, left=t, color=C['flight'], height=0.6)
            t += fl
            last_sim = e['t_sim']
            tm = calls[ci]['timings']
            ci += 1
            dur = ((tm.get('prompt_ms') or 0) + (tm.get('predicted_ms') or 0)) / 1000
            k = e['prompt_id'].rstrip('0123456789').split('_')[0]
            ax.barh(row, dur, left=t, color=C.get(k, '#777'), height=0.6, edgecolor='white', linewidth=0.3)
            t += dur
        end = next((e['t_sim'] for e in ev if e['kind'] == 'done'), last_sim)
        ax.barh(row, max(0, end - last_sim), left=t, color=C['flight'], height=0.6)
    ax.set_yticks(range(len(runs)), [r[1] for r in runs])
    ax.set_xlabel('seconds (flight = sim time, VLM = CPU compute)')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C[k], label=k) for k in ('flight', 'plan', 'survey', 'look', 'triage')],
              frameon=False, ncol=5, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, 1.35))
    ax.grid(axis='y', visible=False)
    save(fig, 'fig_mission_timeline.png')

    # table
    rows = []
    for run, lab, _, m in runs:
        rows.append([lab, f"{m['tp']}/{m['n_gt']}", fmt(m['precision']), fmt(m['recall']), fmt(m['loc_err_survey_m']),
                     fmt(m['loc_err_final_m']), fmt(m['location_acc']), fmt(m['posture_acc']), fmt(m['level_acc']),
                     fmt(m['kendall_tau']), fmt(m['critical_in_topk']), f"{m['n_beacons']}",
                     fmt(m['safe_beacon_rate']), fmt(m.get('replan_correct')), f"{m['n_vlm_calls']} ({m['n_json_errors']})",
                     f"{m['ram_peak_mb'].get('total', 0) / 1024:.1f}"])
    md_table(os.path.join(TAB, 'missions.md'),
             ['run (seed 7)', 'found', 'precision', 'recall', 'loc err survey (m)', 'loc err final (m)', 'roof/water acc',
              'posture acc', 'urgency acc', 'Kendall tau', 'critical in top-k', 'beacons', 'safe beacons', 're-plan correct',
              'VLM calls (JSON err)', 'peak RAM (GB)'], rows)
    # perception table (try_survey)
    rows = []
    for f in sorted(glob.glob(p('runs', 'try_survey_*', 'try_survey.json'))):
        d = json.load(open(f))
        rows.append([os.path.basename(os.path.dirname(f)), str(d['cell']), len(d['gt_in_frame']),
                     fmt(d['survey']['precision']), fmt(d['survey']['recall']), fmt(d['mean_err_survey_m']),
                     fmt(d['mean_err_look_m']), f"{d['json_failures']}/{d['vlm_calls']}"])
    md_table(os.path.join(TAB, 'perception.md'),
             ['run', 'cell (world m)', 'GT people in frame', 'precision', 'recall', 'loc err survey (m)',
              'loc err look-again (m)', 'JSON failures'], rows)


# ───────── flow diagram ──────────────────────────────────────────────────────
def fig_flow(show_run):
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    def img(path, crop=None):
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is None:
            return None
        if im.shape[2] == 4:
            a = im[:, :, 3:] / 255.0
            im = (im[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(16, 8.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 82)
    ax.axis('off')

    def box(x, y, w, h, title, sub='', fc='#f4f6fb', ec='#4a5a80', bold=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.6,rounding_size=1.6', fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h - 2.6, title, ha='center', va='top', fontsize=12.5, weight='bold' if bold else None)
        if sub:
            ax.text(x + w / 2, y + h - 6.3, sub, ha='center', va='top', fontsize=9.5, color='#333', linespacing=1.35)

    def picture(arr, x, y, width):
        """Image centred at (x, y), `width` in axis units (figure is 160 units wide)."""
        if arr is None:
            return
        zoom = width * (16 * 72 / 160) / arr.shape[1]
        ab = AnnotationBbox(OffsetImage(arr, zoom=zoom), (x, y), frameon=True, pad=0.15,
                            bboxprops=dict(ec='#888', lw=0.6))
        ax.add_artist(ab)

    def arrow(x1, y1, x2, y2, txt='', col='#4a5a80', rad=0.0, dy=1.3):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=16, lw=1.6, color=col,
                                     connectionstyle=f'arc3,rad={rad}'))
        if txt:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, txt, ha='center', fontsize=8.8, color=col, style='italic')

    qwen = img(os.path.join(OUT, 'logos', 'qwen_icon.png'))
    QF = '#ece8ff'
    QE = '#6a4fd6'
    # row 1: language
    box(2, 60, 24, 18, 'Responder', '"Find everyone who is\nstranded, judge who needs\nhelp first, mark a safe spot\nfor the rescue boat."')
    box(32, 60, 30, 18, 'Qwen3.5-4B  -  planner         ', 'language reasoning\nJSON tool calls (schema-constrained)\ntakeoff - survey - look_again -\ntriage - mark_beacon - report - home', fc=QF, ec=QE)
    picture(qwen, 59.2, 75.2, 3.6)
    box(68, 60, 22, 18, 'Validator', 'tools, ranges, order,\nsurvey >= 18 m\nfix once, else default plan')
    box(96, 60, 30, 18, 'Mission agent  (ROS 2)', 'executes tools, pauses the sim\nwhile Qwen thinks, re-plans\nwhen /flood/water_level rises')
    arrow(26.5, 69, 31.5, 69)
    arrow(62.5, 69, 67.5, 69)
    arrow(90.5, 69, 95.5, 69)
    # row 2: sim + perception
    box(2, 24, 40, 30, 'Gazebo Harmonic  -  flooded town', 'procedural scenes (seed), rising water,\nEKF + GPS, drone_way_home goal_pose', fc='#eef7f0', ec='#2e7d4f')
    picture(img(os.path.join(OUT, 'sim_oblique.jpg')), 15, 34.5, 23)
    picture(img(os.path.join(OUT, 'uav.jpg')), 34.5, 34.5, 12.5)
    box(48, 24, 30, 30, 'Qwen3.5-4B  -  vision         ', 'grounding in the down camera\n(0-1000 boxes, posture,\nroof / water, beacon point)', fc=QF, ec=QE)
    picture(qwen, 75.2, 51.2, 3.6)
    g = sorted(glob.glob(os.path.join(GRD, '*survey*.jpg')))
    picture(img(g[0]) if g else None, 65.5, 34.5, 18)
    box(84, 24, 30, 30, 'Geometry + memory', 'pixel -> world (K + TF ray-plane)\ntwo-view triangulation:\nroof height above water\nmerge, look-again x3 weight')
    box(120, 24, 38, 30, 'Qwen3.5-4B  -  triage', 'reasons over the memory and\nthe water level:\ncritical / high / medium + why', fc=QF, ec=QE)
    picture(qwen, 155.2, 51.2, 3.6)
    # row 3: outputs
    box(96, 2, 62, 17, 'Beacons, rescue requests, report', 'gz create beacon_<id> on a dry spot   |   /flood/rescue_requests   |\nreport.md + map.png + metrics.json (scored against the scene ground truth)', fc='#fff4e6', ec='#c46a00')
    mp = img(os.path.join(OUT, 'fig_map_seed7.png'))
    if mp is not None:
        mp = mp[int(mp.shape[0] * 0.05):int(mp.shape[0] * 0.93), int(mp.shape[1] * 0.08):int(mp.shape[1] * 0.97)]
    picture(mp, 84, 10.5, 15)
    tri = img(os.path.join(OUT, 'fig_localization.png'))
    picture(tri, 99, 34, 13)
    mem = p('runs', 'seed7_full', 'memory.json')
    if os.path.exists(mem):
        rk = json.load(open(mem))['ranking'][:4]
        for i, r in enumerate(rk):
            col = C.get(r['level'], '#333')
            ax.text(122.5, 40.5 - i * 3.6, f"{i + 1}. {r['person_id']}  {r['level']}", fontsize=9.5, color=col, weight='bold')
            ax.text(135, 40.5 - i * 3.6, r['reason'][:34] + ('...' if len(r['reason']) > 34 else ''), fontsize=8.6,
                    color='#333')
    arrow(111, 59.5, 30, 54.5, 'drone/goal_pose', rad=0.12)
    arrow(42.5, 40, 47.5, 40, 'frame + pose')
    arrow(78.5, 40, 83.5, 40, 'boxes JSON')
    arrow(114.5, 40, 119.5, 40, 'people + heights')
    arrow(139, 23.5, 139, 19.5, 'rescue order')
    arrow(150, 54.5, 120, 59.5, 'water rose -> re-plan', col='#c0392b', rad=-0.15)
    ax.text(80, 80.5, 'FloodScout-VLM: one 4B vision-language model on a laptop CPU (llama.cpp, Q4_K_M) plans, sees and triages',
            ha='center', fontsize=13, weight='bold')
    fig.savefig(os.path.join(OUT, 'fig_flow.png'), dpi=170)
    plt.close(fig)
    print('   fig_flow.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--show-run', default='showcase_seed7')
    args = ap.parse_args()
    for d in (OUT, TAB, GRD):
        os.makedirs(d, exist_ok=True)
    runs = [r for r in ('seed7_full', 'seed7_v1', args.show_run) if os.path.exists(p('runs', r, 'vlm_calls.jsonl'))]
    print('grounding'); fig_grounding(runs)
    print('environment'); fig_environment(args.show_run)
    for r in (args.show_run, 'seed7_full'):
        mp = p('runs', r, 'map.png')
        if os.path.exists(mp):
            import shutil
            shutil.copy(mp, os.path.join(OUT, 'fig_map_seed7.png'))
            break
    print('dataset'); fig_dataset()
    print('vqa'); fig_vqa()
    print('missions'); fig_missions()
    print('flow'); fig_flow(args.show_run)


if __name__ == '__main__':
    main()
