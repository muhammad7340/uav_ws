#!/usr/bin/env python3
"""FloodSim-VQA: frames from the sim + questions with exact answers (FLOOD_VLM_PLAN.md 4.1).

1) collect (needs the seed's world running, water schedule off):
     ros2 launch flood_vlm flood_world.launch.py seed:=3 headless:=true water_schedule:=false &
     ros2 run flood_vlm floodsim_vqa_gen collect --seed 3
   Flies a 3 x 3 grid at 20 m and 12 m; at each point grabs a frame at two water
   levels (0.8 and 2.4 m) and stores the exact camera pose from ground truth.
   scripts/collect_floodsim.sh does this for seeds 1..10.

2) questions (offline):
     ros2 run flood_vlm floodsim_vqa_gen questions --check 20
   Projects GT into every frame with the same camera model as geometry.py and writes
   data/floodsim/questions.jsonl (seeds 9-10 = test split) plus, with --check N,
   N random questions drawn with their GT overlay in data/floodsim/check/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import cv2
import numpy as np

from flood_vlm import geometry as geo
from flood_vlm import gt as G
from flood_vlm import gz_tools
from flood_vlm.config import drone_worlds_dir, load_config
from flood_vlm.perception import TORSO_ABOVE_SURFACE_M

HEAD_ABOVE_SURFACE_M = {'standing': 1.7, 'sitting': 1.0, 'lying': 0.3}
BODY_HALF_M = {'standing': 0.45, 'sitting': 0.5, 'lying': 0.95}
TEST_SEEDS = (9, 10)
WATER_LEVELS = (0.8, 2.4)
GRID = (-18.0, 0.0, 18.0)
ALTS = (20.0, 12.0)


# ───────── collect ───────────────────────────────────────────────────────────
def collect(args):
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from std_msgs.msg import Float32

    from flood_vlm.ros_io import DroneIO

    cfg = load_config()
    out_dir = os.path.join(args.out, 'frames')
    os.makedirs(out_dir, exist_ok=True)
    rclpy.init()
    io = DroneIO(cfg, f'floodsim_collect_{args.seed}')
    pub = io.create_publisher(Float32, '/flood/set_water_level', 10)
    io.start()
    if not io.wait_ready():
        raise SystemExit('no odometry: is the world running?')
    del QoSProfile, DurabilityPolicy

    def set_water(w):
        # Directly through gz (not /flood/set_water_level): the water_controller stopped
        # answering after ~18 rapid changes in a row. Floating debris is not moved.
        for _ in range(3):
            if gz_tools.set_water(w, cfg['world_name']):
                io.water = float(w)
                pub.publish(Float32(data=float(w)))      # keep /flood/water_level consumers in sync
                time.sleep(0.5)                          # let the render catch up
                return
            time.sleep(1.0)
        raise RuntimeError(f'water did not change to {w}')

    rows = []
    set_water(WATER_LEVELS[0])
    for alt in ALTS:
        pts = [(x, y) for y in GRID for x in (GRID if GRID.index(y) % 2 == 0 else GRID[::-1])]
        for i, (x, y) in enumerate(pts):
            io.goto_world(x, y, alt)
            levels = WATER_LEVELS if i % 2 == 0 else WATER_LEVELS[::-1]   # fewer water changes
            for w in levels:
                if abs(io.water - w) > 1e-3:
                    set_water(w)
                f = io.grab(settle_s=1.0)
                if f is None:
                    continue
                R, t = geo.camera_pose_from_base(io.gt, io.gt_quat)      # exact pose for GT projection
                name = f's{args.seed:02d}_a{int(alt)}_w{w:.1f}_{i}.jpg'
                cv2.imwrite(os.path.join(out_dir, name), f.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append({'seed': args.seed, 'image': os.path.abspath(os.path.join(out_dir, name)),
                             'alt': alt, 'water': w, 'cell': [x, y], 'K': f.K.tolist(),
                             'R': R.tolist(), 't': t.tolist(), 'size': list(f.size)})
                print(f'{name}', flush=True)
    with open(os.path.join(args.out, 'frames.jsonl'), 'a') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    set_water(WATER_LEVELS[0])
    io.goto_world(cfg['home']['x'], cfg['home']['y'], 20.0)
    io.stop()
    rclpy.try_shutdown()
    print(f'{len(rows)} frames for seed {args.seed}')


# ───────── projection helpers ────────────────────────────────────────────────
class Cam:
    def __init__(self, fr):
        self.K = np.array(fr['K'])
        self.R = np.array(fr['R'])
        self.t = np.array(fr['t'])
        self.w, self.h = fr['size']

    def px(self, p):
        return geo.world_to_pixel(p, self.K, self.R, self.t)

    def inside(self, uv, margin=8):
        return uv is not None and margin <= uv[0] < self.w - margin and margin <= uv[1] < self.h - margin

    def rel(self, uv):
        return [int(round(v)) for v in geo.pixel_to_rel(uv[0], uv[1], self.w, self.h)]


def person_box(cam: Cam, p: dict) -> list[float] | None:
    """Pixel bbox of a person: a box of +-BODY_HALF around the torso, projected."""
    z0 = p['z']
    hz = z0 + HEAD_ABOVE_SURFACE_M[p['posture']]
    r = BODY_HALF_M[p['posture']]
    pts = [cam.px((p['x'] + dx, p['y'] + dy, z)) for dx in (-r, r) for dy in (-r, r) for z in (z0, hz)]
    pts = [q for q in pts if q is not None]
    if not pts:
        return None
    us, vs = [q[0] for q in pts], [q[1] for q in pts]
    return [min(us), min(vs), max(us), max(vs)]


def visible_people(cam: Cam, gt: dict, w: float) -> list[dict]:
    out = []
    for p in gt['people']:
        if p['z'] + HEAD_ABOVE_SURFACE_M[p['posture']] <= w + 0.1:       # under water
            continue
        uv = cam.px((p['x'], p['y'], p['z'] + TORSO_ABOVE_SURFACE_M[p['posture']]))
        if cam.inside(uv, 12):
            out.append(dict(p, uv=uv, box=person_box(cam, p)))
    return out


def sample_in_polygon(poly, rng, avoid):
    xs, ys = [q[0] for q in poly], [q[1] for q in poly]
    for _ in range(50):
        x, y = rng.uniform(min(xs), max(xs)), rng.uniform(min(ys), max(ys))
        if G.point_in_polygon(x, y, poly) and not any(G.point_in_polygon(x, y, a) for a in avoid):
            return x, y
    return None


# ───────── questions ─────────────────────────────────────────────────────────
def frame_questions(fr: dict, gt: dict, rng: random.Random, img_dir: str) -> list[dict]:
    cam = Cam(fr)
    w = fr['water']
    vis = visible_people(cam, gt, w)
    qs = []

    def add(qtype, q, a, atype, **kw):
        qs.append({'type': qtype, 'question': q, 'answer': a, 'answer_type': atype, **kw})

    # A head within 0.3 m of the surface is faintly visible through the water: counts in
    # such a frame are ambiguous, so they are not asked.
    def near_surface(p):
        head = p['z'] + HEAD_ABOVE_SURFACE_M[p['posture']]
        uv = cam.px((p['x'], p['y'], w))
        return abs(head - w) < 0.3 and cam.inside(uv, 0)
    ambiguous = any(near_surface(p) for p in gt['people'])

    # simple counting
    if not ambiguous:
        add('simple_counting', 'How many people are visible in this image?', str(len(vis)), 'int')
    # complex counting
    n_water = sum(p['support'] == 'water' for p in vis)
    n_roof = sum(p['support'] == 'roof' for p in vis)
    if not ambiguous and (n_water or rng.random() < 0.3):
        add('complex_counting', 'How many people are standing in flood water?', str(n_water), 'int')
    if not ambiguous and n_roof and rng.random() < 0.5:
        add('complex_counting', 'How many people are on rooftops?', str(n_roof), 'int')
    houses = [h for h in gt['houses'] if cam.inside(cam.px((h['x'], h['y'], h['roof_z'])), 0)]
    if houses and rng.random() < 0.5:
        n = sum(w > h['plinth_z'] + 0.05 for h in houses)
        add('complex_counting', 'How many houses have flood water above their ground floor level?', str(n), 'int')
    # condition: a road point, embankments (dry) half of the time when visible
    # anything that covers a ground-level street from above: houses, mounds, embankments
    blocks = [h['polygon'] for h in gt['houses']] + [z['polygon'] for z in gt['dry_zones'] if z['type'] != 'roof']
    roads = [r for r in gt['roads']]
    rng.shuffle(roads)
    roads.sort(key=lambda r: (not r['embankment']) if rng.random() < 0.5 else r['embankment'])
    for r in roads:
        xy = sample_in_polygon(r['polygon'], rng, blocks if not r['embankment'] else [])
        if xy is None:
            continue
        uv = cam.px((xy[0], xy[1], r['top_z']))
        if not cam.inside(uv, 30):
            continue
        ans = 'passable' if r['top_z'] > w + 0.05 else 'flooded'
        add('condition', f'Is the road at point {cam.rel(uv)} (0-1000 image coordinates) passable or flooded?',
            ans, 'label', options=['passable', 'flooded'], road=r['id'])
        break
    # yes / no
    if vis:
        c = min(vis, key=lambda p: math.hypot(p['uv'][0] - cam.w / 2, p['uv'][1] - cam.h / 2))
        dry = c['support'] == 'roof' and G.roof_margin(gt, c, w) > 0
        add('yes_no', 'Is the person nearest the centre of the image on a dry roof?', 'yes' if dry else 'no',
            'yes_no', person=c['id'])
    if not ambiguous and rng.random() < 0.5:
        add('yes_no', 'Is anyone standing in the flood water in this image?', 'yes' if n_water else 'no', 'yes_no')
    # grounding
    lying = [p for p in vis if p['posture'] == 'lying']
    if len(lying) == 1 and lying[0]['box']:
        add('grounding', 'Point to the person lying down.', [round(v, 1) for v in lying[0]['box']], 'point',
            person=lying[0]['id'])
    wading = [p for p in vis if p['support'] == 'water']
    if len(wading) == 1 and wading[0]['box']:
        add('grounding', 'Point to the person standing in the flood water.', [round(v, 1) for v in wading[0]['box']],
            'point', person=wading[0]['id'])
    # priority: two marked people with different GT urgency
    rank = {r['id']: (G.LEVEL_RANK[r['level']], r['margin']) for r in G.ranking(gt, w)}
    pairs = [(a, b) for i, a in enumerate(vis) for b in vis[i + 1:]
             if a['box'] and b['box'] and (rank[a['id']][0] != rank[b['id']][0]
                                           or abs(rank[a['id']][1] - rank[b['id']][1]) >= 0.5)]
    if pairs:
        a, b = rng.choice(pairs)
        if rng.random() < 0.5:
            a, b = b, a
        img = cv2.imread(fr['image'])
        for lbl, p, col in (('A', a, (0, 0, 255)), ('B', b, (255, 0, 0))):
            x1, y1, x2, y2 = [int(v) for v in p['box']]
            cv2.rectangle(img, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), col, 2)
            cv2.putText(img, lbl, (x1 - 4, max(12, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        name = os.path.splitext(os.path.basename(fr['image']))[0] + f'_AB_{a["id"]}_{b["id"]}.jpg'
        path = os.path.join(img_dir, name)
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        first = 'A' if rank[a['id']] < rank[b['id']] else 'B'
        add('priority', 'Two people are marked A and B (boxes drawn). Who needs the rescue boat first? '
            'Answer A or B.', first, 'label', options=['A', 'B'], image=os.path.abspath(path),
            pair=[a['id'], b['id']])
    for q in qs:
        q.setdefault('image', fr['image'])
        q.update({'seed': fr['seed'], 'alt': fr['alt'], 'water': w})
    return qs


def draw_check(q: dict, fr: dict, gt: dict, path: str):
    cam = Cam(fr)
    img = cv2.imread(q['image'])
    for p in visible_people(cam, gt, fr['water']):
        x1, y1, x2, y2 = [int(v) for v in p['box']]
        col = (0, 0, 255) if p['support'] == 'water' else (0, 200, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 1)
        cv2.putText(img, f"{p['id']} {p['posture'][:3]}", (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    if q['answer_type'] == 'point':
        x1, y1, x2, y2 = [int(v) for v in q['answer']]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    if 'point [' in q['question']:
        import re
        px, py = map(int, re.search(r'point \[(\d+), (\d+)\]', q['question']).groups())
        u, v = geo.rel_to_pixel(px, py, cam.w, cam.h)
        cv2.circle(img, (int(u), int(v)), 6, (255, 0, 255), 2)
    pad = np.full((54, img.shape[1], 3), 255, np.uint8)
    cv2.putText(pad, q['question'][:95], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    cv2.putText(pad, f"answer: {q['answer']}   water {fr['water']} m  alt {fr['alt']} m", (4, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 160), 1)
    cv2.imwrite(path, np.vstack([img, pad]))


def questions(args):
    frames = [json.loads(line) for line in open(os.path.join(args.out, 'frames.jsonl'))]
    img_dir = os.path.join(args.out, 'marked')
    os.makedirs(img_dir, exist_ok=True)
    gts = {}
    rng = random.Random(0)
    out = []
    by_image = {}
    for fr in frames:
        s = fr['seed']
        if s not in gts:
            gts[s] = G.load_gt(os.path.join(drone_worlds_dir(), f'flood_town_{s}.gt.json'))
        by_image[fr['image']] = fr
        for q in frame_questions(fr, gts[s], rng, img_dir):
            q['frame'] = fr['image']
            q['qid'] = f'fs{len(out)}'
            q['dataset'] = 'floodsim'
            q['split'] = 'test' if s in TEST_SEEDS else 'train'
            out.append(q)
    path = os.path.join(args.out, 'questions.jsonl')
    with open(path, 'w') as f:
        for q in out:
            f.write(json.dumps(q) + '\n')
    from collections import Counter
    print(f'{len(out)} questions from {len(frames)} frames -> {path}')
    print('by type:', dict(Counter(q['type'] for q in out)))
    print('by split:', dict(Counter(q['split'] for q in out)))
    if args.check:
        cdir = os.path.join(args.out, 'check')
        os.makedirs(cdir, exist_ok=True)
        for q in random.Random(1).sample(out, min(args.check, len(out))):
            fr = by_image[q['frame']]
            draw_check(q, fr, gts[fr['seed']], os.path.join(cdir, f"{q['qid']}_{q['type']}.jpg"))
        print(f'{args.check} overlays in {cdir}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--seed', type=int, required=True)
    c.add_argument('--out', default='data/floodsim')
    q = sub.add_parser('questions')
    q.add_argument('--out', default='data/floodsim')
    q.add_argument('--check', type=int, default=0, help='draw N random questions with GT overlays')
    args = ap.parse_args(argv)
    collect(args) if args.cmd == 'collect' else questions(args)


if __name__ == '__main__':
    main()
