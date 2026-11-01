"""End-to-end mission logic without Gazebo or llama-server.

A fake drone teleports to goals and renders nothing; a fake VLM answers from the
scene ground truth through the same camera model, so the whole pipeline
(plan -> survey -> look again -> triage -> water rise -> re-plan -> beacons ->
report) runs in a second.
"""
import json
import math
import os

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')

from flood_vlm import geometry as geo  # noqa: E402
from flood_vlm import gt as G  # noqa: E402
from flood_vlm import gz_tools, mission_node  # noqa: E402
from flood_vlm.config import load_config  # noqa: E402
from flood_vlm.ros_io import Frame  # noqa: E402
from flood_vlm.scene_gen import SceneGen  # noqa: E402

W, H, HFOV = 640, 480, 1.3962634


class _Pub:
    def publish(self, msg):
        # serialize like a real publisher: catches wrong field types (int colours, ...)
        from rclpy.serialization import serialize_message
        serialize_message(msg)


class _Log:
    def info(self, *_):
        pass

    warn = error = info


class _Clock:
    def __init__(self, io):
        self.io = io

    def now(self):
        class _T:
            def __init__(s, t):
                s.nanoseconds = int(t * 1e9)

            def to_msg(s):
                from builtin_interfaces.msg import Time
                return Time(sec=int(s.nanoseconds // 10**9), nanosec=int(s.nanoseconds % 10**9))
        return _T(self.io.t)


class FakeIO:
    def __init__(self, cfg, gt, rise_at):
        self.cfg, self.gt = cfg, gt
        self.t = 0.0
        self.water = cfg['water']['start']
        self.water_listeners = []
        self.ow = geo.OdomWorld((cfg['home']['x'], cfg['home']['y'], gt['home']['z'] + 0.3))
        self.pos = np.array([cfg['home']['x'], cfg['home']['y'], gt['home']['z'] + 0.3])
        self.path_world = []
        self.rise_at = list(rise_at)       # flight seconds at which the water steps up
        self.K = geo.intrinsics_from_hfov(W, H, HFOV)

    def get_logger(self):
        return _Log()

    def get_clock(self):
        return _Clock(self)

    def create_publisher(self, *_a, **_k):
        return _Pub()

    def now(self):
        return self.t

    def pause(self, _p):
        return True

    def sleep_sim(self, s, wall_timeout=None):
        self._advance(s)

    def _advance(self, dt):
        self.t += dt
        steps = self.cfg['water']['steps']
        while self.rise_at and self.t >= self.rise_at[0]:
            self.rise_at.pop(0)
            self.water = next((w for w in steps if w > self.water), self.water)
            for f in self.water_listeners:
                f(self.water)

    def goto_world(self, x, y, z, timeout_s=None):
        d = math.dist(self.pos, (x, y, z))
        self.pos = np.array([x, y, z], float)
        self.path_world.append((self.t, x, y, z))
        self._advance(d / 2.0 + 2.0)
        return True

    def drone_world(self):
        return self.pos

    def grab(self, settle_s=None, timeout=60.0):
        R, t = geo.camera_pose_from_base(self.pos, (0, 0, 0, 1))
        return Frame(image=np.zeros((H, W, 3), np.uint8), K=self.K, R=R, t=t, stamp=self.t,
                     drone_world=self.pos.copy(), pose_source='fake', water=self.water)


class FakeVLM:
    """Answers from GT: people in view get boxes, look-again picks a beacon 2.5 m towards home."""

    def __init__(self, io, gt, cfg):
        self.io, self.gt, self.cfg = io, gt, cfg
        self.calls = []

    def _visible(self, frame):
        from flood_vlm.perception import TORSO_ABOVE_SURFACE_M
        out = []
        for p in self.gt['people']:
            torso = p['z'] + TORSO_ABOVE_SURFACE_M[p['posture']]
            uv = geo.world_to_pixel((p['x'], p['y'], torso), frame.K, frame.R, frame.t)
            if uv and 10 < uv[0] < W - 10 and 10 < uv[1] < H - 10:
                out.append((p, uv))
        return out

    def ask(self, image, prompt, schema, think=False, max_tokens=384, prompt_id='', meta=None, **_):
        self.calls.append(prompt_id)
        loc = {'roof': 'on_roof', 'water': 'in_water'}
        if prompt_id.startswith('plan'):
            out = {'steps': [{'tool': 'takeoff', 'args': {'alt': 20}},
                             {'tool': 'survey', 'args': {'alt': 20, 'cells': [[-19, -13], [0, -13], [19, -13], [19, 13], [0, 13], [-19, 13]]}},
                             {'tool': 'return_home', 'args': {}}]}
        elif prompt_id.startswith('survey'):
            frame = self.io.grab()
            people = []
            for p, (u, v) in self._visible(frame):
                x, y = geo.pixel_to_rel(u, v, W, H)
                people.append({'bbox_2d': [int(x - 15), int(y - 25), int(x + 15), int(y + 25)],
                               'location': loc[p['support']], 'posture': 'unsure', 'confidence': 0.5})
            out = {'people': people, 'dry_areas': [], 'roads': [], 'flooded_fraction': 0.7}
        elif prompt_id.startswith('look'):
            frame = self.io.grab()
            vis = self._visible(frame)
            if not vis:
                out = {'person_present': False, 'bbox_2d': [0, 0, 1, 1], 'location': 'on_roof',
                       'posture': 'unsure', 'surface_dry': False, 'beacon_point_2d': [0, 0],
                       'reason': '', 'confidence': 0.1}
            else:
                import re
                px, py = map(int, re.search(r'point \[(\d+), (\d+)\]', prompt).groups())
                tu, tv = geo.rel_to_pixel(px, py, W, H)
                p, (u, v) = min(vis, key=lambda q: math.hypot(q[1][0] - tu, q[1][1] - tv))
                # roof: 2 m towards the roof centre; water: centre of the nearest dry zone
                if p['house']:
                    h = next(q for q in self.gt['houses'] if q['id'] == p['house'])
                    tx, ty = h['x'], h['y']
                    d = math.hypot(tx - p['x'], ty - p['y']) or 1.0
                    bxw = (p['x'] + 2.0 * (tx - p['x']) / d, p['y'] + 2.0 * (ty - p['y']) / d, p['z'])
                else:
                    zs = [z for z in self.gt['dry_zones'] if z['type'] != 'roof']
                    cen = lambda z: (sum(q[0] for q in z['polygon']) / 4, sum(q[1] for q in z['polygon']) / 4)  # noqa: E731
                    z = min(zs, key=lambda z: math.dist(cen(z), (p['x'], p['y'])))
                    bxw = (*cen(z), z['top_z'])
                bu, bv = geo.world_to_pixel(bxw, frame.K, frame.R, frame.t)
                assert bu is not None
                x, y = geo.pixel_to_rel(u, v, W, H)
                out = {'person_present': True, 'bbox_2d': [int(x - 30), int(y - 50), int(x + 30), int(y + 50)],
                       'location': loc[p['support']], 'posture': p['posture'],
                       'surface_dry': p['support'] == 'roof',
                       'beacon_point_2d': [int(v) for v in geo.pixel_to_rel(bu, bv, W, H)],
                       'reason': 'flat dry roof next to the water', 'confidence': 0.9}
        elif prompt_id.startswith('triage'):
            brief = json.loads(prompt.split('):\n', 1)[1].split('\n\nRank', 1)[0])
            w = self.io.water

            def lvl(b):
                if b['location'] == 'in_water' or b['posture'] == 'lying':
                    return 'critical'
                return 'high' if (b['roof_height_est'] or 99) - w < 1.0 else 'medium'
            rk = sorted(brief, key=lambda b: (G.LEVEL_RANK[lvl(b)], b['height_above_water_est'] if b['height_above_water_est'] is not None else 99))
            out = {'ranking': [{'person_id': b['person_id'], 'level': lvl(b), 'reason': 'test'} for b in rk]}
        else:
            out = {'error': f'unexpected prompt {prompt_id}'}
        return out, {'wall_s': 0.01, 'prompt_n': 10, 'predicted_n': 10}


def test_full_mission_offline(tmp_path, monkeypatch):
    cfg = load_config()
    gen = SceneGen(7, cfg)
    gen.build(overview=False)
    gt = gen.gt
    gt_path = tmp_path / 'flood_town_7.gt.json'
    gt_path.write_text(json.dumps(gt))
    monkeypatch.setattr(mission_node, 'drone_worlds_dir', lambda: str(tmp_path))
    # first rise during the survey, second right after the first beacon goes down
    io = FakeIO(cfg, gt, rise_at=[60.0])

    def spawn(*_a, **_k):
        if not io.rise_at and io.water < max(cfg['water']['steps']):
            io.rise_at.append(io.t + 1.0)
        return True
    monkeypatch.setattr(gz_tools, 'spawn_beacon', spawn)
    m = mission_node.Mission(io, cfg, {'seed': 7, 'lookagain': True, 'runs_root': str(tmp_path),
                                       'run_id': 'offline'})
    m.vlm = FakeVLM(io, gt, cfg)
    m.ask = lambda image, prompt, schema, think, mt, pid, meta=None: m.vlm.ask(image, prompt, schema, think, mt, pid)
    m.run(mission_node.DEFAULT_MISSION)

    rd = tmp_path / 'offline'
    metrics = json.loads((rd / 'metrics.json').read_text())
    assert (rd / 'report.md').exists() and (rd / 'map.png').exists()
    assert metrics['plan_source'] == 'vlm' and metrics['plan_valid_first_try']
    assert metrics['recall'] == 1.0, metrics
    assert metrics['precision'] == 1.0, metrics
    assert metrics['loc_err_final_m'] < 0.5
    assert metrics['n_beacons'] == len(gt['people'])
    assert metrics['safe_beacon_rate'] >= 0.7, metrics['beacon_checks']
    assert metrics['n_replans'] >= 1, 'water rose during the beacon phase: expected a re-plan'
    assert metrics['replan_correct'] == 1.0, m.replans
    roof = [p for p in m.mem.active() if p.location == 'on_roof']
    assert sum(p.surface_z is not None for p in roof) >= len(roof) - 1   # triangulated
    assert metrics['critical_in_topk'] == 1.0
    assert io.water == max(cfg['water']['steps'])
    assert 'return_home' not in m.vlm.calls
    assert os.path.exists(rd / 'events.jsonl')
