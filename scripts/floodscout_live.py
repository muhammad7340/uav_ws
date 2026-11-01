#!/usr/bin/env python3
"""Live FloodScout dashboard window for a mission that is running now.

    python3 scripts/floodscout_live.py --run runs/<run_id>      # started by scripts/floodscout_demo.sh

Reads the run folder as the mission writes it (events, model calls, recorded camera frames)
and shows the same layout as the demo video. Keys: q / Esc = close, f = toggle fullscreen.
When the mission ends it switches to the results screen.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from floodscout_dashboard import BG, SUB, TXT, Dashboard, H, RunData, W, font  # noqa: E402

WIN = 'FloodScout'


def waiting(msg):
    img = Image.fromarray(np.full((H, W, 3), BG, np.uint8))
    d = ImageDraw.Draw(img)
    d.text((140, 440), 'FloodScout', font=font(76, True), fill=TXT)
    d.text((144, 545), msg, font=font(30), fill=SUB)
    return np.array(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--fps', type=float, default=6.0)
    args = ap.parse_args()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)   # no Qt toolbar
    cv2.resizeWindow(WIN, 1600, 900)
    dash = Dashboard()
    rd = None
    full = False
    intro_until = None
    last_t, last_wall = None, time.time()
    while True:
        t_loop = time.time()
        if rd is None and os.path.exists(os.path.join(args.run, 'events.jsonl')):
            rd = RunData(args.run)
        if rd is None:
            frame = waiting('Starting the simulation and the model ...')
        else:
            rd.reload()
            mpath = os.path.join(args.run, 'metrics.json')
            if rd.done and os.path.exists(mpath):
                try:
                    frame = dash.results(rd, json.load(open(mpath)))
                except (json.JSONDecodeError, OSError):
                    frame = waiting('Writing the report ...')
            elif rd.t0 is None:
                # before take-off: the request, then the planning card
                if intro_until is None and rd.mission:
                    intro_until = time.time() + 4.0
                if intro_until and time.time() < intro_until:
                    frame = dash.intro(rd, 1.0 - (intro_until - time.time()) / 3.5)
                else:
                    cards = rd.cards()
                    frame = dash.mission(rd, 0.0, cards[-1] if cards else None, 1.0, thinking=not cards)
            else:
                # sim time = newest overview frame; it stops while the simulation is paused (model thinking)
                t_new = rd.show.latest_t
                if t_new != last_t:
                    last_t, last_wall = t_new, time.time()
                ts = max(0.0, (t_new or rd.t0) - rd.t0)
                frame = dash.mission(rd, ts, None, 1.0, thinking=time.time() - last_wall > 2.0)
        cv2.imshow(WIN, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(max(1, int(1000 / args.fps - (time.time() - t_loop) * 1000))) & 0xFF
        # stop on q / Esc or when the window was really closed (-1); minimizing it must not stop the mission
        if key in (ord('q'), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_AUTOSIZE) < 0:
            break
        if key == ord('f'):
            full = not full
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
