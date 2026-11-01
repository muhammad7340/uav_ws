#!/usr/bin/env python3
"""Render the ~90 s FloodScout demo video from a recorded run.

    source scripts/flood_env.sh && scripts/record_showcase.sh 7      # record (~30 min)
    python3 scripts/make_video.py --run runs/showcase_seed7          # render (~2 min)

Same layout as the live window (scripts/floodscout_dashboard.py). Flight is sped up; every
model call becomes a short card (the simulation is paused while the model thinks).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from floodscout_dashboard import WS, Dashboard, H, RunData, W  # noqa: E402

FPS = 30
INTRO, OUTRO = 4.5, 6.0
HOLD = {'plan': 4.5, 'survey': 2.2, 'look': 1.4, 'triage': 3.5}


class Video:
    def __init__(self, run, flight_seconds=48.0):
        self.rd = RunData(run)
        self.dash = Dashboard()
        self.metrics = json.load(open(os.path.join(run, 'metrics.json')))
        end = next((e['t_sim'] for e in self.rd.events if e['kind'] == 'done'), self.rd.events[-1]['t_sim'])
        self.end = end
        self.speed = max(1.0, end / flight_seconds)
        tl, v, last = [], INTRO, 0.0
        for ts, kind, call, e in self.rd.cards():
            v += (ts - last) / self.speed
            tl.append((v, ts, None))
            tl.append((v + HOLD.get(kind, 1.5), ts, (ts, kind, call, e)))
            v += HOLD.get(kind, 1.5)
            last = ts
        v += (end - last) / self.speed
        tl.append((v, end, None))
        self.tl = tl
        self.duration = v + OUTRO

    def at(self, tv):
        prev_v, prev_s = INTRO, 0.0
        for v, s, card in self.tl:
            if tv <= v:
                if card is not None:
                    return s, card, (tv - prev_v) / max(1e-6, v - prev_v)
                f = (tv - prev_v) / (v - prev_v) if v > prev_v else 1.0
                return prev_s + f * (s - prev_s), None, 1.0
            prev_v, prev_s = v, s
        return self.end, None, 1.0

    def frame(self, tv):
        if tv < INTRO:
            return self.dash.intro(self.rd, (tv - 0.5) / 3.0)
        if tv > self.duration - OUTRO:
            return self.dash.results(self.rd, self.metrics)
        ts, card, prog = self.at(tv)
        return self.dash.mission(self.rd, ts, card, prog, thinking=card is not None and prog < 0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=os.path.join(WS, 'runs', 'showcase_seed7'))
    ap.add_argument('--out', default=os.path.join(WS, 'assets', 'floodscout', 'floodscout_demo.mp4'))
    args = ap.parse_args()
    v = Video(args.run)
    n = int(v.duration * FPS)
    print(f'{v.duration:.1f} s video, flight x{v.speed:.1f}, {len(v.rd.show.files)} overview frames')
    tmp = args.out + '.raw.mp4'
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    for i in range(n):
        wr.write(cv2.cvtColor(v.frame(i / FPS), cv2.COLOR_RGB2BGR))
        if i % (FPS * 15) == 0:
            print(f'  {i / FPS:5.1f} s', flush=True)
    wr.release()
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', tmp, '-c:v', 'libx264', '-preset', 'slow', '-crf', '22',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', args.out], check=True)
    os.remove(tmp)
    for k, tv in enumerate((3.5, v.duration * 0.25, v.duration * 0.55, v.duration - 2)):
        cv2.imwrite(os.path.join(os.path.dirname(args.out), f'video_still_{k}.jpg'),
                    cv2.cvtColor(v.frame(tv), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print('wrote', args.out, f'{os.path.getsize(args.out) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
