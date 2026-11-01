#!/usr/bin/env python3
"""FloodScout dashboard: one 1920x1080 layout shared by the live window and the demo video.

    RunData(run_dir)  -> everything a mission writes (events, model calls, camera frames, path)
    Dashboard().mission(state) / .intro(...) / .results(...) -> RGB frame

Left: Gazebo overview camera (town, drone + trail, water, beacons), the drone's down camera and
the mission plan. Right: the current vision-language model call (role, image with its detections, answer),
then the latest mission events.
"""
from __future__ import annotations

import bisect
import glob
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WS = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(WS, 'src', 'flood_vlm'))
sys.path.insert(0, os.path.join(WS, 'scripts'))
from flood_vlm import geometry as geo  # noqa: E402
from make_paper_figures import draw_boxes  # noqa: E402

W, H = 1920, 1080
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
BG = (20, 22, 28)
PANEL = (32, 35, 44)
CARD = (40, 44, 55)
LINE = (62, 66, 80)
ACC = (150, 125, 245)       # Qwen purple
OK = (100, 200, 130)
WARN = (240, 170, 70)
RED = (235, 95, 85)
TXT = (232, 234, 240)
SUB = (175, 180, 195)
DIM = (125, 130, 145)
LEVEL = {'critical': RED, 'high': WARN, 'medium': (100, 165, 240)}
ROLE = {'plan': 'Mission planning', 'survey': 'Visual survey (20 m)', 'look': 'Close inspection (8 m)',
        'triage': 'Triage reasoning'}
KIND = {'beacon': 'Beacon', 'replan': 'Re-plan', 'water': 'Water', 'triage': 'Triage', 'look': 'Inspection',
        'home': 'Return', 'survey': 'Survey', 'plan': 'Plan', 'done': 'Done'}
_fonts = {}


def font(size, bold=False):
    k = (size, bold)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(BOLD if bold else FONT, size)
    return _fonts[k]


def jl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:      # a line still being written
                pass
    return out


def wrap(text, f, width):
    words, lines, cur = str(text).split(), [], ''
    for w in words:
        test = (cur + ' ' + w).strip()
        if f.getlength(test) <= width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def cap(s):
    s = str(s)
    return s[:1].upper() + s[1:]


def nice_msg(kind, msg):
    """Mission log line -> short readable sentence for the dashboard."""
    import re
    m = msg
    m = m.replace('on_roof', 'on a roof').replace('in_water', 'in water').replace('on_dry_ground', 'on dry ground')
    m = re.sub(r'; beacon \[.*$', '', m)
    m = re.sub(r' \(no ranking yet\)', '', m)
    m = re.sub(r'beacon at \(([-\d.]+), ([-\d.]+), [-\d.]+\)', r'beacon placed at (\1, \2)', m)
    m = re.sub(r'^Cell (\d+) \(([-\d]+), ([-\d]+)\): (\d+) people, (\d+) dry areas, flooded ([\d.]+)',
               lambda g: f"Image {int(g[1]) + 1} of the survey: {g[4]} people, {g[5]} dry areas", cap(m))
    m = re.sub(r' at \(([-\d.]+), ([-\d.]+)\)$', r' at (\1, \2) m', m)
    m = m.replace('order unchanged', 'rescue order unchanged').replace(' -> rejected', ', rejected')
    return cap(m)


def kind_of(prompt_id):
    return prompt_id.rstrip('0123456789').split('_')[0]


def step_text(step):
    """A plan step in plain words (methodology, not function names)."""
    t, a = step.get('tool', ''), step.get('args', {}) or {}
    alt = a.get('alt')
    pid = a.get('person_id')
    who = 'every person found' if pid in (None, 'all') else pid
    return {
        'takeoff': f"Take off to {alt:g} m" if isinstance(alt, (int, float)) else 'Take off',
        'survey': (f"Survey the area from {alt:g} m" if isinstance(alt, (int, float)) else 'Survey the area')
                  + (f" ({len(a['cells'])} image positions)" if a.get('cells') else ''),
        'look_again': f'Inspect {who} at close range',
        'triage': 'Rank people by urgency',
        'mark_beacon': f'Mark a safe beacon for {who}',
        'report': 'Write the rescue report',
        'return_home': 'Return to base and land',
    }.get(t, cap(t.replace('_', ' ')))


class Frames:
    """Images named by sim time; latest at or before t. rescan() picks up new files (live)."""

    def __init__(self, folder):
        self.folder = folder
        self.files, self.t = [], []
        self._cache = (None, None)
        self.rescan()

    def rescan(self):
        self.files = sorted(glob.glob(os.path.join(self.folder, '*.jpg')))
        self.t = [float(os.path.basename(f)[:-4]) for f in self.files]

    def at(self, t):
        if not self.files:
            return None
        i = max(0, bisect.bisect_right(self.t, t) - 1)
        if self._cache[0] != self.files[i]:
            img = cv2.imread(self.files[i])
            if img is None:                          # half-written file
                return self._cache[1]
            self._cache = (self.files[i], img)
        return self._cache[1]

    @property
    def latest_t(self):
        return self.t[-1] if self.t else None


class RunData:
    """A mission run folder (runs/<id>), complete or still being written."""

    def __init__(self, run):
        self.run = run
        rec = os.path.join(run, 'rec')
        self.show = Frames(os.path.join(rec, 'showcase'))
        self.down = Frames(os.path.join(rec, 'down'))
        self.cams = None
        self.reload()

    def reload(self):
        rec = os.path.join(self.run, 'rec')
        self.show.rescan()
        self.down.rescan()
        if self.cams is None and os.path.exists(os.path.join(rec, 'cameras.json')):
            self.cams = json.load(open(os.path.join(rec, 'cameras.json')))
            c = self.cams['showcase']
            rl = geo.rpy_to_matrix(0.0, c['pitch'], c['yaw'])
            self.R = rl @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
            self.tcam = np.array(c['eye'], float)
            self.K = np.array(c['K'], float)
        self.events = jl(os.path.join(self.run, 'events.jsonl'))
        self.calls = jl(os.path.join(self.run, 'vlm_calls.jsonl'))
        path = os.path.join(rec, 'path.csv')
        try:
            self.path = np.loadtxt(path, delimiter=',', ndmin=2) if os.path.getsize(path) else np.zeros((0, 4))
        except (OSError, ValueError):
            self.path = np.zeros((0, 4))
        wpath = os.path.join(rec, 'water.csv')
        try:
            self.water = np.loadtxt(wpath, delimiter=',', ndmin=2) if os.path.getsize(wpath) else np.zeros((0, 2))
        except (OSError, ValueError):
            self.water = np.zeros((0, 2))
        self.mission = next((e['msg'].split(': ', 1)[1].strip('"') for e in self.events if e['kind'] == 'start'), '')
        self.plan = self.parse_plan()
        self.t0 = self.mission_t0()
        self.done = any(e['kind'] == 'done' for e in self.events)

    def parse_plan(self):
        """Executed plan from the 'plan' event: "takeoff{...} -> survey{...} -> ..."."""
        ev = [e for e in self.events if e['kind'] == 'plan' and ' -> ' in e['msg']]
        if not ev:
            return []
        steps = []
        for part in ev[-1]['msg'].split(' -> '):
            name, _, args = part.partition('{')
            try:
                a = json.loads('{' + args) if args else {}
            except json.JSONDecodeError:
                a = {}
            steps.append({'tool': name.strip(), 'args': a})
        return steps

    def mission_t0(self):
        """Absolute sim time of mission t=0: the drone leaves the pad right after 'takeoff'."""
        if len(self.path) < 2:
            return None
        z0 = self.path[0, 3]
        lift = next((r[0] for r in self.path if r[3] > z0 + 0.25), None)
        if lift is None:
            return None
        takeoff = next((e['t_sim'] for e in self.events if e['kind'] == 'tool' and e['msg'].startswith('takeoff')), 0.0)
        return lift - takeoff - 0.6

    def water_at(self, ts):
        if len(self.water) and self.t0 is not None:
            i = np.searchsorted(self.water[:, 0], self.t0 + ts, side='right') - 1
            if i >= 0:
                return float(self.water[i, 1])
        lvl = 0.8
        for e in self.events:
            if e['t_sim'] > ts:
                break
            try:
                if e['kind'] == 'triage' and 'water ' in e['msg']:
                    lvl = float(e['msg'].split('water ')[1].split(' m')[0])
                elif e['kind'] in ('replan', 'water') and 'rose to ' in e['msg']:
                    lvl = float(e['msg'].split('rose to ')[1].split(' m')[0])
            except (IndexError, ValueError):
                pass
        return lvl

    def cards(self):
        """[(t_sim, kind, call, event)] for every model call, in order."""
        out = []
        it = iter(self.calls)
        for e in self.events:
            if e['kind'] != 'vlm' or 'thinking failed' in e['msg']:
                continue
            pid = e.get('prompt_id', '')
            call = next((c for c in it if c['prompt_id'].startswith(pid)), None)
            out.append((e['t_sim'], kind_of(pid), call, e))
        return out

    def project(self, xyz):
        return geo.world_to_pixel(xyz, self.K, self.R, self.tcam) if self.cams else None


def paste(canvas, img, x, y, w, h):
    canvas[y:y + h, x:x + w] = cv2.cvtColor(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


class Dashboard:
    def __init__(self):
        logo = os.path.join(WS, 'assets', 'floodscout', 'logos', 'qwen_icon.png')
        self.logo = Image.open(logo).convert('RGBA').resize((40, 40)) if os.path.exists(logo) else None

    # ── main mission screen ───────────────────────────────────────────────────
    def mission(self, rd: RunData, ts: float, card=None, prog: float = 1.0, thinking: bool = False):
        img = np.full((H, W, 3), BG, np.uint8)
        t_abs = (rd.t0 or 0.0) + ts
        sc = rd.show.at(t_abs if rd.t0 is not None else (rd.show.latest_t or 0))
        if sc is not None:
            sc = sc.copy()
            if rd.t0 is not None and len(rd.path):
                trail = rd.path[rd.path[:, 0] <= t_abs]
                pts = [rd.project(q) for q in trail[::4, 1:4]]
                pts = [(int(a), int(b)) for a, b in (q for q in pts if q is not None)]
                for a, b in zip(pts, pts[1:]):
                    cv2.line(sc, a, b, (90, 215, 125), 2, cv2.LINE_AA)
                if len(trail):
                    uv = rd.project(trail[-1, 1:4])
                    if uv is not None:
                        cv2.circle(sc, (int(uv[0]), int(uv[1])), 15, (70, 70, 245), 3, cv2.LINE_AA)
            paste(img, sc, 16, 72, 1264, 711)
        dn = rd.down.at(t_abs if rd.t0 is not None else (rd.down.latest_t or 0))
        if dn is not None:
            paste(img, dn, 16, 799, 376, 265)
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        self.header(d, ts, rd.water_at(ts))
        d.rounded_rectangle([22, 805, 40 + font(15, True).getlength('Down camera'), 829], 5, fill=(20, 22, 28))
        d.text((30, 808), 'Down camera', font=font(15, True), fill=TXT)
        self.plan_panel(d, rd, ts)
        self.model_panel(pil, d, rd, ts, card, prog, thinking)
        return np.array(pil)

    def header(self, d, ts, water):
        d.rectangle([0, 0, W, 60], fill=PANEL)
        d.text((24, 13), 'FloodScout', font=font(28, True), fill=TXT)
        d.text((210, 21), 'Language-guided UAV flood-relief scouting with a vision-language model',
               font=font(18), fill=SUB)
        d.text((1340, 21), 'Sim time', font=font(16), fill=DIM)
        d.text((1420, 17), f'{ts:4.0f} s', font=font(21, True), fill=TXT)
        d.text((1560, 21), 'Water level', font=font(16), fill=DIM)
        d.rounded_rectangle([1660, 16, 1900, 44], 6, outline=LINE, width=2)
        d.rounded_rectangle([1662, 18, 1662 + int(236 * min(1.0, water / 3.0)), 42], 5, fill=(120, 96, 62))
        d.text((1672, 19), f'{water:.1f} m', font=font(18, True), fill=TXT)

    def plan_panel(self, d, rd, ts):
        x, y = 404, 791
        d.rounded_rectangle([x, y, 1280, H - 16], 10, fill=PANEL)
        d.text((x + 18, y + 12), 'Mission plan', font=font(19, True), fill=TXT)
        d.text((x + 160, y + 15), "generated by the VLM planner from the responder's request", font=font(15), fill=DIM)
        tools = [e for e in rd.events if e['kind'] == 'tool' and e['t_sim'] <= ts + 1e-6]
        cur = len(tools) - 1
        for i, st in enumerate(rd.plan[:8]):
            yy = y + 48 + i * 28
            done, now = i < cur, i == cur
            col = ACC if now else (OK if done else DIM)
            mark = '▶' if now else ('✓' if done else '·')
            d.text((x + 20, yy), mark, font=font(17, True), fill=col)
            d.text((x + 48, yy), f'{i + 1}.  {step_text(st)}', font=font(18, now), fill=TXT if now else col)
        last = [e for e in rd.events if e['t_sim'] <= ts + 1e-6 and e['kind'] not in ('vlm', 'tool', 'plan', 'start')]
        if last:
            e = last[-1]
            d.line([x + 18, H - 54, 1262, H - 54], fill=LINE)
            d.text((x + 18, H - 44), f"{KIND.get(e['kind'], cap(e['kind']))}:", font=font(15, True), fill=SUB)
            d.text((x + 120, H - 44), nice_msg(e['kind'], e['msg'])[:95], font=font(15), fill=DIM)

    def model_panel(self, pil, d, rd, ts, card, prog, thinking):
        x0, x1 = 1296, W - 16
        d.rounded_rectangle([x0, 72, x1, H - 16], 10, fill=PANEL)
        if self.logo:
            pil.paste(self.logo, (x0 + 18, 86), self.logo)
        d.text((x0 + 70, 90), 'Qwen', font=font(26, True), fill=ACC)
        d.text((x0 + 82 + font(26, True).getlength('Qwen'), 95), '3.5-4B vision-language model', font=font(20), fill=SUB)
        shown = card
        if shown is None:
            past = [c for c in rd.cards() if c[0] <= ts + 1e-6]
            shown = past[-1] if past else None
        y = 146
        if thinking:
            d.rounded_rectangle([x0 + 16, y, x1 - 16, y + 40], 8, fill=(58, 48, 96))
            d.text((x0 + 30, y + 9), 'Processing the next request …  simulation paused',
                   font=font(17), fill=TXT)
            y += 54
        if thinking and shown is not None:
            d.text((x0 + 20, y - 6), 'Previous answer', font=font(14, True), fill=DIM)
            y += 16
        if shown is None:
            d.text((x0 + 20, y + 4), 'Waiting for the first request …', font=font(19), fill=DIM)
            self.event_log(d, rd, ts, x0, max(y + 40, 880))
            return
        _, kind, call, e = shown
        tm = (call or {}).get('timings', {})
        secs = ((tm.get('prompt_ms') or 0) + (tm.get('predicted_ms') or 0)) / 1000
        d.text((x0 + 20, y), ROLE.get(kind, cap(kind)), font=font(21, True), fill=TXT)
        d.text((x0 + 20, y + 30), f'{secs:.0f} s  ·  {tm.get("predicted_n", "–")} output tokens',
               font=font(15), fill=DIM)
        y += 64
        out = (call or {}).get('output', {})
        tw = x1 - x0 - 40
        if call and call.get('image') and os.path.exists(call['image']) and kind in ('survey', 'look'):
            im = cv2.imread(call['image'])
            if prog > 0.3:
                im = draw_boxes(im, out)
            iw = tw
            ih = int(iw * 3 / 4)
            pil.paste(Image.fromarray(cv2.cvtColor(cv2.resize(im, (iw, ih)), cv2.COLOR_BGR2RGB)), (x0 + 20, y))
            y += ih + 14
            lines = self.describe(kind, out)
            for line in lines:
                for wl in wrap(line, font(17), tw)[:3]:
                    d.text((x0 + 20, y), wl, font=font(17), fill=TXT)
                    y += 24
        elif kind == 'plan':
            d.text((x0 + 20, y), 'Request', font=font(15, True), fill=DIM)
            y += 22
            for line in wrap(f'“{rd.mission}”', font(18), tw)[:4]:
                d.text((x0 + 20, y), line, font=font(18), fill=TXT)
                y += 26
            reasoning = out.get('reasoning', '')
            if reasoning:
                y += 10
                d.text((x0 + 20, y), 'Reasoning', font=font(15, True), fill=DIM)
                y += 22
                n = int(len(reasoning) * min(1.0, prog * 1.6))
                for line in wrap(cap(reasoning[:max(1, n)]), font(17), tw)[:7]:
                    d.text((x0 + 20, y), line, font=font(17), fill=ACC)
                    y += 24
            y += 10
            d.text((x0 + 20, y), 'Planned actions', font=font(15, True), fill=DIM)
            y += 24
            for i, st in enumerate(out.get('steps', [])[:8]):
                d.text((x0 + 20, y), f'{i + 1}.  {step_text(st)}', font=font(17), fill=TXT)
                y += 25
        elif kind == 'triage':
            w = rd.water_at(e['t_sim'])
            d.text((x0 + 20, y), f'Reasoning at water level {w:.1f} m', font=font(15, True), fill=DIM)
            y += 22
            for line in wrap(cap(out.get('reasoning', '')), font(16), tw)[:6]:
                d.text((x0 + 20, y), line, font=font(16), fill=ACC)
                y += 22
            y += 10
            d.text((x0 + 20, y), 'Rescue order', font=font(15, True), fill=DIM)
            y += 24
            seen = set()
            k = 0
            for r in out.get('ranking', []):
                if r['person_id'] in seen or k >= 9:
                    continue
                seen.add(r['person_id'])
                k += 1
                d.text((x0 + 20, y), f"{k}.", font=font(17, True), fill=SUB)
                d.text((x0 + 50, y), r['person_id'], font=font(17, True), fill=TXT)
                d.rounded_rectangle([x0 + 96, y + 1, x0 + 186, y + 23], 5, fill=LEVEL.get(r['level'], DIM))
                d.text((x0 + 104, y + 2), cap(r['level']), font=font(15, True), fill=(20, 20, 24))
                d.text((x0 + 198, y + 2), cap(r['reason'])[:42], font=font(15), fill=SUB)
                y += 28
        self.event_log(d, rd, ts, x0, max(y + 18, 880))

    @staticmethod
    def describe(kind, out):
        if kind == 'survey':
            ppl = out.get('people', [])
            if not ppl:
                head = 'No people detected.'
            else:
                n = len(ppl)
                det = '; '.join(f"{p['posture']} {p['location'].replace('_', ' ').replace('on roof', 'on a roof')}" for p in ppl)
                head = f"{n} {'person' if n == 1 else 'people'} detected: {det}."
            ff = out.get('flooded_fraction')
            tail = f"{len(out.get('dry_areas', []))} dry areas found" + (f"; about {ff * 100:.0f} % of the frame is flooded." if isinstance(ff, (int, float)) else '.')
            return [head, tail]
        if not out.get('person_present'):
            return ['No person at this location: the detection is discarded.']
        loc = str(out.get('location', '')).replace('_', ' ').replace('on roof', 'on a roof')
        return [f"Person confirmed: {out.get('posture')}, {loc}; surface {'dry' if out.get('surface_dry') else 'wet'}.",
                f"Beacon: {cap(out.get('reason', ''))}"]

    @staticmethod
    def event_log(d, rd, ts, x0, y):
        d.line([x0 + 18, y - 10, W - 34, y - 10], fill=LINE)
        d.text((x0 + 20, y - 2), 'Mission events', font=font(15, True), fill=DIM)
        y += 24
        evs = [e for e in rd.events if e['t_sim'] <= ts + 1e-6
               and e['kind'] in ('beacon', 'replan', 'water', 'triage', 'look', 'home', 'done')][-6:]
        for ev in evs:
            col = {'replan': RED, 'water': RED, 'beacon': WARN, 'home': OK, 'done': OK}.get(ev['kind'], SUB)
            d.text((x0 + 20, y), f"{ev['t_sim']:4.0f} s", font=font(14), fill=DIM)
            d.text((x0 + 82, y), KIND.get(ev['kind'], cap(ev['kind'])), font=font(14, True), fill=col)
            d.text((x0 + 186, y), nice_msg(ev['kind'], ev['msg'])[:54], font=font(14), fill=SUB)
            y += 21
            if y > H - 36:
                break

    # ── intro and results screens ─────────────────────────────────────────────
    def intro(self, rd: RunData, typed: float):
        img = np.full((H, W, 3), BG, np.uint8)
        sc = rd.show.at(rd.t0 or 0.0) if rd.show.files else None
        if sc is not None:
            paste(img, sc, 0, 0, W, H)
            img[:] = (img * 0.25 + np.array(BG) * 0.35).astype(np.uint8)
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        d.text((140, 220), 'FloodScout', font=font(76, True), fill=TXT)
        d.text((144, 322), 'Language-guided UAV flood-relief scouting with a vision-language model',
               font=font(30), fill=SUB)
        d.text((144, 440), 'Responder request', font=font(24, True), fill=ACC)
        q = f'“{rd.mission}”'
        n = int(len(q) * min(1.0, max(0.0, typed)))
        y = 486
        for line in wrap(q[:n], font(40), 1600):
            d.text((144, y), line, font=font(40), fill=TXT)
            y += 58
        return np.array(pil)

    def results(self, rd: RunData, metrics: dict):
        img = np.full((H, W, 3), BG, np.uint8)
        mp = cv2.imread(os.path.join(rd.run, 'map.png'))
        if mp is not None:
            mp = mp[36:]                             # drop the map's own title line
            h = 960
            w = int(mp.shape[1] * h / mp.shape[0])
            paste(img, mp, 60, 60, w, h)
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        m = metrics
        x = 1130
        d.text((x, 90), 'Mission results', font=font(40, True), fill=TXT)
        d.text((x, 148), 'Scored against the ground truth of the simulated scene', font=font(20), fill=SUB)
        vlm_s = (m.get('prompt_ms', 0) + m.get('gen_ms', 0)) / 1000
        rows = [
            ('People found', f"{m.get('tp')} of {m.get('n_gt')}"),
            ('Precision', f"{m.get('precision', 0):.2f}"),
            ('Localization error', f"{m.get('loc_err_survey_m') or 0:.2f} m  →  {m.get('loc_err_final_m') or 0:.2f} m"),
            ('Safe beacon placements', f"{round((m.get('safe_beacon_rate') or 0) * (m.get('n_beacons') or 0))} of {m.get('n_beacons')}"),
            ('Urgency level accuracy', f"{m.get('level_acc') or 0:.2f}"),
            ('Re-planning after water rise', f"{m.get('n_replans')} " + ('event' if m.get('n_replans') == 1 else 'events')),
            ('Model calls', f"{m.get('n_vlm_calls')}  ({m.get('n_json_errors')} malformed)"),
            ('Flight time / model time', f"{m.get('flight_sim_s', 0):.0f} s  /  {vlm_s:.0f} s"),
        ]
        y = 220
        for k, v in rows:
            d.text((x, y), k, font=font(24), fill=SUB)
            d.text((x + 380, y), v, font=font(24, True), fill=TXT)
            y += 50
            d.line([x, y - 12, W - 80, y - 12], fill=LINE)
        y += 24
        d.text((x, y), 'Map legend', font=font(20, True), fill=SUB)
        y += 34
        for t in ('Numbers give the rescue order chosen by the model.', 'Stars mark the placed beacons.',
                  'Hollow circles mark the true positions of the people.'):
            d.text((x, y), t, font=font(20), fill=DIM)
            y += 30
        return np.array(pil)
