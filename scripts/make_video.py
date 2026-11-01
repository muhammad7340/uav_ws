#!/usr/bin/env python3
"""Render the ~90 s FloodScout demo video from a recorded run.

    source scripts/flood_env.sh && scripts/record_showcase.sh 7      # record (~30 min)
    python3 scripts/make_video.py --run runs/showcase_seed7          # render (~2 min)

Left: Gazebo showcase camera (whole town, drone marker, rising water, beacons), the live
down camera and the mission plan with the current step. Right: what Qwen3.5-4B is doing:
the call (plan / survey / look-again / triage), the image it saw with its boxes, its answer
and its compute time; then the rescue ranking and an event log. Flight is sped up; every VLM
call becomes a short card (the sim is paused while Qwen thinks, so no flight is lost).
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WS = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(WS, 'src', 'flood_vlm'))
sys.path.insert(0, os.path.join(WS, 'scripts'))
from flood_vlm import geometry as geo  # noqa: E402
from make_paper_figures import draw_boxes  # noqa: E402

W, H, FPS = 1920, 1080, 30
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
BG = (24, 26, 33)
PANEL = (36, 39, 50)
ACC = (130, 100, 240)        # Qwen purple (RGB)
OK = (90, 200, 120)
WARN = (240, 160, 60)
RED = (230, 80, 70)
TXT = (230, 232, 240)
DIM = (150, 155, 170)
LEVEL = {'critical': RED, 'high': WARN, 'medium': (90, 160, 240)}
HOLD = {'plan': 4.5, 'survey': 2.2, 'look': 1.4, 'triage': 3.5}
_fonts = {}


def font(size, bold=False):
    k = (size, bold)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(BOLD if bold else FONT, size)
    return _fonts[k]


def jl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


class Frames:
    """Images named by sim time; latest at or before t."""

    def __init__(self, folder):
        self.files = sorted(glob.glob(os.path.join(folder, '*.jpg')))
        self.t = [float(os.path.basename(f)[:-4]) for f in self.files]
        self._cache = (None, None)

    def at(self, t):
        if not self.files:
            return None
        i = max(0, bisect.bisect_right(self.t, t) - 1)
        if self._cache[0] != i:
            self._cache = (i, cv2.imread(self.files[i]))
        return self._cache[1]


def wrap(text, f, width):
    words, lines, cur = text.split(), [], ''
    for w in words:
        test = (cur + ' ' + w).strip()
        if f.getlength(test) <= width:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def paste(canvas, img, x, y, w, h):
    canvas[y:y + h, x:x + w] = cv2.cvtColor(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


class Video:
    def __init__(self, run):
        self.run = run
        rec = os.path.join(run, 'rec')
        self.show = Frames(os.path.join(rec, 'showcase'))
        self.down = Frames(os.path.join(rec, 'down'))
        self.cams = json.load(open(os.path.join(rec, 'cameras.json')))
        self.path = np.loadtxt(os.path.join(rec, 'path.csv'), delimiter=',', ndmin=2)
        self.events = jl(os.path.join(run, 'events.jsonl'))
        self.calls = jl(os.path.join(run, 'vlm_calls.jsonl'))
        self.mem = json.load(open(os.path.join(run, 'memory.json')))
        self.metrics = json.load(open(os.path.join(run, 'metrics.json')))
        self.mission = self.events[0]['msg'].split(': ', 1)[1].strip('"')
        self.plan = self.mem['plan']['plan']['steps']
        # mission t_sim=0 in absolute sim time: the drone leaves the pad right after takeoff
        z0 = self.path[0, 3]
        lift = next((r[0] for r in self.path if r[3] > z0 + 0.25), self.path[0, 0])
        takeoff = next((e['t_sim'] for e in self.events if e['kind'] == 'tool' and e['msg'].startswith('takeoff')), 0.0)
        self.t0 = lift - takeoff - 0.6
        c = self.cams['showcase']
        rl = geo.rpy_to_matrix(0.0, c['pitch'], c['yaw'])
        self.R = rl @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
        self.tcam = np.array(c['eye'], float)
        self.K = np.array(c['K'], float)
        self.build_timeline()

    # ── timeline: list of (video_seconds, sim_time, card or None) ─────────────
    def build_timeline(self, flight_seconds=48.0):
        vlm = [e for e in self.events if e['kind'] == 'vlm' and 'thinking failed' not in e['msg']]
        call_iter = iter(self.calls)
        cards = []
        for e in vlm:
            pid = e.get('prompt_id', '')
            call = next((c for c in call_iter if c['prompt_id'].startswith(pid)), None)
            cards.append((e['t_sim'], pid, call, e))
        end = next((e['t_sim'] for e in self.events if e['kind'] == 'done'), self.events[-1]['t_sim'])
        self.end_sim = end
        speed = max(1.0, end / flight_seconds)
        tl = []
        v = 4.0                                      # intro
        last = 0.0
        for ts, pid, call, e in cards:
            v += (ts - last) / speed
            kind = pid.rstrip('0123456789').split('_')[0]
            tl.append((v, ts, None))
            tl.append((v + HOLD.get(kind, 1.5), ts, (kind, pid, call, e)))
            v += HOLD.get(kind, 1.5)
            last = ts
        v += (end - last) / speed
        tl.append((v, end, None))
        self.speed = speed
        self.cards = tl
        self.duration = v + 6.0                      # outro

    def at(self, tv):
        """(sim_time, active card or None, card progress 0..1) at video time tv (after the intro)."""
        prev_v, prev_s = 4.0, 0.0
        for v, s, card in self.cards:
            if tv <= v:
                if card is not None:
                    hold = v - prev_v
                    return s, card, (tv - prev_v) / hold if hold else 1.0
                frac = (tv - prev_v) / (v - prev_v) if v > prev_v else 1.0
                return prev_s + frac * (s - prev_s), None, 0.0
            prev_v, prev_s = v, s
        return self.end_sim, None, 0.0

    # ── drawing ───────────────────────────────────────────────────────────────
    def drone_px(self, t_abs):
        i = min(len(self.path) - 1, max(0, np.searchsorted(self.path[:, 0], t_abs)))
        return geo.world_to_pixel(self.path[i, 1:4], self.K, self.R, self.tcam), self.path[:i + 1, 1:4]

    def last_events(self, ts, kinds=None, n=7):
        evs = [e for e in self.events if e['t_sim'] <= ts + 1e-6 and (kinds is None or e['kind'] in kinds)]
        return evs[-n:]

    def frame(self, tv):
        img = np.full((H, W, 3), BG, np.uint8)
        if tv < 4.0:
            return self.intro(img, tv)
        if tv > self.duration - 6.0:
            return self.outro(img, tv - (self.duration - 6.0))
        ts, card, prog = self.at(tv)
        t_abs = self.t0 + ts
        # left: showcase
        sc = self.show.at(t_abs)
        if sc is not None:
            sc = sc.copy()
            uv, trail = self.drone_px(t_abs)
            pts = [geo.world_to_pixel(q, self.K, self.R, self.tcam) for q in trail[::4]]
            pts = [(int(a), int(b)) for a, b in (q for q in pts if q is not None)]
            for a, b in zip(pts, pts[1:]):
                cv2.line(sc, a, b, (80, 220, 120), 2, cv2.LINE_AA)
            if uv is not None:
                cv2.circle(sc, (int(uv[0]), int(uv[1])), 16, (60, 60, 255), 3, cv2.LINE_AA)
            paste(img, sc, 0, 60, 1280, 720)
        dn = self.down.at(t_abs)
        if dn is not None:
            paste(img, dn, 0, 790, 386, 290)
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        self.header(d, ts)
        self.plan_panel(d, ts, 396, 790)
        self.qwen_panel(pil, d, ts, card, prog)
        return np.array(pil)

    def header(self, d, ts):
        d.rectangle([0, 0, W, 58], fill=PANEL)
        d.text((18, 12), 'FloodScout', font=font(28, True), fill=TXT)
        d.text((300, 18), 'Qwen3.5-4B plans, sees and triages a flood rescue  (ROS 2 + Gazebo)',
               font=font(20), fill=DIM)
        water = next((float(e['msg'].split('water ')[1].split(' m')[0]) for e in reversed(self.last_events(ts, ('triage',), 50))
                      if 'water ' in e['msg']), None)
        wl = self.water_at(ts)
        d.text((1450, 16), f'sim t = {ts:5.0f} s', font=font(20, True), fill=TXT)
        d.rectangle([1640, 14, 1900, 44], outline=DIM)
        d.rectangle([1640, 14, 1640 + int(260 * min(1, wl / 3.0)), 44], fill=(120, 95, 60))
        d.text((1652, 17), f'water {wl:.1f} m', font=font(19, True), fill=TXT)
        del water

    def water_at(self, ts):
        lvl = 0.8
        for e in self.events:
            if e['t_sim'] > ts:
                break
            if e['kind'] in ('triage', 'replan', 'water') and 'water' in e['msg']:
                try:
                    lvl = float(e['msg'].split('water ')[1].split(' m')[0]) if e['kind'] == 'triage' else \
                        float(e['msg'].split('rose to ')[1].split(' m')[0])
                except (IndexError, ValueError):
                    pass
        return lvl

    def plan_panel(self, d, ts, x, y):
        d.rectangle([x, y, 1280, H], fill=PANEL)
        d.text((x + 14, y + 10), 'Plan from Qwen (validated JSON tool calls)', font=font(19, True), fill=TXT)
        tools = [e for e in self.events if e['kind'] == 'tool' and e['t_sim'] <= ts + 1e-6]
        cur = len(tools) - 1
        for i, s in enumerate(self.plan):
            yy = y + 44 + i * 30
            args = ''
            if s['args']:
                a = dict(s['args'])
                if 'cells' in a:
                    a['cells'] = f"{len(a['cells'])} cells"
                args = ', '.join(f'{k}={v}' for k, v in a.items())
            col = ACC if i == cur else (OK if i < cur else DIM)
            mark = '>' if i == cur else ('v' if i < cur else ' ')
            d.text((x + 16, yy), f'{mark} {i + 1}. {s["tool"]}', font=font(19, i == cur), fill=col)
            if args:
                d.text((x + 260, yy), args, font=font(17), fill=DIM)
        d.text((x + 16, H - 34), 'status: ' + (self.last_events(ts, None, 1)[0]['msg'][:78] if self.last_events(ts, None, 1) else ''),
               font=font(16), fill=DIM)

    def qwen_panel(self, pil, d, ts, card, prog):
        x0 = 1290
        d.rectangle([x0, 60, W, H], fill=PANEL)
        logo = os.path.join(WS, 'assets', 'floodscout', 'logos', 'qwen_icon.png')
        if os.path.exists(logo):
            lg = Image.open(logo).convert('RGBA').resize((44, 44))
            pil.paste(lg, (x0 + 14, 70), lg)
        d.text((x0 + 68, 76), 'Qwen3.5-4B  (llama.cpp, Q4_K_M)', font=font(22, True), fill=ACC)
        shown = card
        if shown is None:
            past = [c for v, s, c in self.cards if c is not None and s <= ts]
            shown = past[-1] if past else None
        if shown is None:
            d.text((x0 + 16, 140), 'waiting for the first call ...', font=font(20), fill=DIM)
            return
        kind, pid, call, e = shown
        tm = (call or {}).get('timings', {})
        secs = ((tm.get('prompt_ms') or 0) + (tm.get('predicted_ms') or 0)) / 1000
        role = {'plan': 'language planner', 'survey': 'vision: survey 20 m', 'look': 'vision: look-again 8 m',
                'triage': 'language reasoning: triage'}.get(kind, kind)
        d.text((x0 + 16, 128), f'{role}', font=font(21, True), fill=TXT)
        d.text((x0 + 16, 158), f'{secs:.0f} s compute   |   {tm.get("prompt_n", "?")} prompt tok, '
               f'{tm.get("predicted_n", "?")} out   |   sim paused', font=font(16), fill=DIM)
        y = 192
        out = (call or {}).get('output', {})
        if call and call.get('image') and os.path.exists(call['image']) and kind in ('survey', 'look'):
            im = cv2.imread(call['image'])
            if card is None or prog > 0.35:
                im = draw_boxes(im, out)
            arr = cv2.cvtColor(cv2.resize(im, (600, 450)), cv2.COLOR_BGR2RGB)
            pil.paste(Image.fromarray(arr), (x0 + 15, y))
            y += 462
            if kind == 'survey':
                ppl = out.get('people', [])
                txt = f'{len(ppl)} people: ' + ', '.join(f"{p['posture']} {p['location'].replace('_', ' ')}" for p in ppl)
                txt += f".  {len(out.get('dry_areas', []))} dry areas, flooded {out.get('flooded_fraction', '?')}"
            else:
                txt = (f"person: {out.get('posture')} {str(out.get('location', '')).replace('_', ' ')}, surface dry "
                       f"{out.get('surface_dry')}; beacon at {out.get('beacon_point_2d')} - {out.get('reason', '')}"
                       if out.get('person_present') else 'no person here -> rejected')
            for line in wrap(txt, font(17), 600)[:4]:
                d.text((x0 + 16, y), line, font=font(17), fill=TXT)
                y += 24
        elif kind == 'plan':
            d.text((x0 + 16, y), 'query:', font=font(18, True), fill=DIM)
            y += 28
            for line in wrap(f'"{self.mission}"', font(18), 600):
                d.text((x0 + 16, y), line, font=font(18), fill=TXT)
                y += 26
            y += 10
            reasoning = out.get('reasoning', '')
            if reasoning:
                d.text((x0 + 16, y), 'reasoning:', font=font(18, True), fill=DIM)
                y += 28
                n = int(len(reasoning) * min(1.0, prog * 1.6 + (0 if card else 1)))
                for line in wrap(reasoning[:max(1, n)], font(17), 600)[:7]:
                    d.text((x0 + 16, y), line, font=font(17), fill=ACC)
                    y += 24
            y += 10
            d.text((x0 + 16, y), 'plan:', font=font(18, True), fill=DIM)
            y += 28
            for s in out.get('steps', [])[:8]:
                d.text((x0 + 16, y), f"- {s.get('tool')} {json.dumps(s.get('args', {}))[:52]}", font=font(16), fill=TXT)
                y += 23
        elif kind == 'triage':
            d.text((x0 + 16, y), f"water {e.get('meta', {}).get('water', '') or self.water_at(ts):.1f} m - reasoning:",
                   font=font(18, True), fill=DIM)
            y += 28
            for line in wrap(out.get('reasoning', ''), font(17), 600)[:6]:
                d.text((x0 + 16, y), line, font=font(17), fill=ACC)
                y += 24
            y += 10
            for i, r in enumerate(out.get('ranking', [])[:10]):
                d.text((x0 + 16, y), f"{i + 1}. {r['person_id']}", font=font(18, True), fill=TXT)
                d.text((x0 + 90, y), r['level'], font=font(18, True), fill=LEVEL.get(r['level'], TXT))
                d.text((x0 + 190, y), r['reason'][:44], font=font(16), fill=DIM)
                y += 27
        # event log
        y = max(y + 16, 880)
        d.line([x0 + 10, y - 8, W - 10, y - 8], fill=(70, 74, 90))
        for ev in self.last_events(ts, ('beacon', 'replan', 'water', 'triage', 'look', 'home'), 6):
            col = {'replan': RED, 'water': RED, 'beacon': WARN, 'home': OK}.get(ev['kind'], DIM)
            d.text((x0 + 16, y), f"{ev['t_sim']:5.0f}s {ev['kind']}: {ev['msg'][:58]}", font=font(15), fill=col)
            y += 22
            if y > H - 22:
                break

    def intro(self, img, tv):
        sc = self.show.at(self.t0)
        if sc is not None:
            paste(img, sc, 0, 0, W, H)
            img[:] = (img * 0.28).astype(np.uint8)
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        d.text((120, 200), 'FloodScout', font=font(72, True), fill=TXT)
        d.text((124, 300), 'a 4B vision-language model plans, sees and triages a flood rescue',
               font=font(30), fill=DIM)
        q = f'"{self.mission}"'
        n = int(len(q) * min(1.0, max(0.0, (tv - 0.6) / 2.6)))
        y = 440
        d.text((124, y - 50), 'responder:', font=font(28, True), fill=ACC)
        for line in wrap(q[:n], font(40), 1650):
            d.text((124, y), line, font=font(40), fill=TXT)
            y += 56
        return np.array(pil)

    def outro(self, img, t):
        m = self.metrics
        mp = cv2.imread(os.path.join(self.run, 'map.png'))
        if mp is not None:
            mp = mp[36:]            # drop the title line baked into older run maps
            h = 1000
            w = int(mp.shape[1] * h / mp.shape[0])
            paste(img, mp, 40, 50, w, h)
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        x = 1090
        d.text((x, 80), 'Mission result vs. scene ground truth', font=font(28, True), fill=TXT)
        rows = [('people found', f"{m.get('tp')}/{m.get('n_gt')}  (precision {m.get('precision', 0):.2f})"),
                ('position error', f"{m.get('loc_err_survey_m') or 0:.2f} m -> {m.get('loc_err_final_m') or 0:.2f} m (look-again)"),
                ('safe beacons', f"{m.get('safe_beacon_rate') or 0:.2f}  ({m.get('n_beacons')} placed)"),
                ('re-plans on water rise', f"{m.get('n_replans')}" + (f"  (correct {m['replan_correct']:.0%})" if m.get('replan_correct') is not None else '')),
                ('VLM calls', f"{m.get('n_vlm_calls')}  ({m.get('n_json_errors')} JSON errors)"),
                ('flight / VLM time', f"{m.get('flight_sim_s')} s / {m.get('prompt_ms', 0) / 1000 + m.get('gen_ms', 0) / 1000:.0f} s"),
                ('peak RAM', f"{(m.get('ram_peak_mb') or {}).get('total', 0) / 1024:.1f} GB")]
        y = 150
        for k, v in rows:
            d.text((x, y), k, font=font(24), fill=DIM)
            d.text((x + 290, y), v, font=font(23, True), fill=TXT)
            y += 52
        d.text((x, y + 30), 'numbers = rescue order, stars = beacons,', font=font(22), fill=DIM)
        d.text((x, y + 62), 'hollow circles = ground truth people', font=font(22), fill=DIM)
        d.text((x, H - 90), 'github.com/muhammad7340/uav_ws  -  src/flood_vlm', font=font(22), fill=ACC)
        return np.array(pil)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=os.path.join(WS, 'runs', 'showcase_seed7'))
    ap.add_argument('--out', default=os.path.join(WS, 'assets', 'floodscout', 'floodscout_demo.mp4'))
    args = ap.parse_args()
    v = Video(args.run)
    n = int(v.duration * FPS)
    print(f'{v.duration:.1f} s video, flight x{v.speed:.1f}, {len(v.show.files)} showcase frames')
    tmp = args.out + '.raw.mp4'
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    for i in range(n):
        wr.write(cv2.cvtColor(v.frame(i / FPS), cv2.COLOR_RGB2BGR))
        if i % (FPS * 10) == 0:
            print(f'  {i / FPS:5.1f} s', flush=True)
    wr.release()
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', tmp, '-c:v', 'libx264', '-preset', 'slow', '-crf', '22',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', args.out], check=True)
    os.remove(tmp)
    # a few stills for the README
    for k, tv in enumerate((2.5, v.duration * 0.25, v.duration * 0.55, v.duration - 3)):
        cv2.imwrite(os.path.join(os.path.dirname(args.out), f'video_still_{k}.jpg'),
                    cv2.cvtColor(v.frame(tv), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print('wrote', args.out, f'{os.path.getsize(args.out) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
