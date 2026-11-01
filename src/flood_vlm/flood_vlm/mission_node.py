#!/usr/bin/env python3
"""FloodScout-VLM mission agent (FLOOD_VLM_PLAN.md 3.1).

    ros2 run flood_vlm mission_node --ros-args -p seed:=7 -p mission:="Find everyone ..."

Plan (VLM, think on) -> validated tool calls -> survey -> look again -> triage
(think on) -> beacons in triage order -> report -> return home. When
/flood/water_level rises, triage is re-run at the next safe point and the
remaining beacons are re-ordered.

Ground truth (*.gt.json) is read only by the scorer at the end, never for decisions.
ROS callbacks run on DroneIO's executor thread; this node's main thread runs the
mission and blocks on VLM calls (the sim is paused while the VLM thinks).
"""
from __future__ import annotations

import json
import math
import os
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, String
from visualization_msgs.msg import Marker, MarkerArray

from flood_vlm import gz_tools, prompts, schemas
from flood_vlm import gt as G
from flood_vlm import report as R
from flood_vlm.config import drone_worlds_dir, load_config
from flood_vlm.memory import ObjectMemory
from flood_vlm.perception import look_position, look_prompt, process_look, process_survey
from flood_vlm.ros_io import DroneIO
from flood_vlm.validator import default_plan, normalize, validate
from flood_vlm.vlm_client import VLMClient

DEFAULT_MISSION = ('Find everyone who is stranded, judge who needs help first, and mark a safe '
                   'spot near each person for the rescue boat.')
LEVEL_RGB = {'critical': (0.9, 0.1, 0.1), 'high': (1.0, 0.6, 0.0), 'medium': (0.2, 0.6, 1.0),
             None: (0.6, 0.6, 0.6)}
BEACON_HOVER_ABOVE_M = 6.0
BEACON_HOVER_S = 3.0
BEACON_PUSH_M = 2.5          # beacon distance when the VLM's point is too close
LOOK_MIN_ABOVE_TARGET_M = 4.0


class Mission:
    def __init__(self, io: DroneIO, cfg: dict, args: dict):
        self.io = io
        self.cfg = cfg
        self.args = args
        self.log = io.get_logger()
        stamp = time.strftime('%Y%m%d_%H%M%S')
        self.run_id = args.get('run_id') or f'seed{args["seed"]}_{stamp}'
        self.run_dir = os.path.join(args.get('runs_root', 'runs'), self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        llm = cfg['llm']
        self.vlm = VLMClient(llm['url'], run_dir=self.run_dir, temperature=llm['temperature'],
                             image_long_side=args.get('image_long_side'))
        self.mem = ObjectMemory(cfg['merge_radius_m'], self.run_dir)
        self.ranking: list[dict] = []
        self.ranking_water = None
        self.beacons: list[dict] = []
        self.replans: list[dict] = []
        self.events: list[dict] = []
        self.plan_info: dict = {}
        self.t_vlm = 0.0
        self.t_flight_sim = 0.0
        self.t_flight_wall = 0.0
        self.water_rose = threading.Event()
        self.water_log: list[tuple[float, float]] = [(0.0, io.water)]
        io.water_listeners.append(self._on_water)
        self.req_pub = io.create_publisher(String, '/flood/rescue_requests', 10)
        self.marker_pub = io.create_publisher(MarkerArray, '/flood/markers', 10)
        self.event_pub = io.create_publisher(String, '/flood/events', 10)
        self.start_pub = io.create_publisher(Empty, '/flood/mission_started', 10)
        self.t0_sim = None

    # ── utilities ─────────────────────────────────────────────────────────────
    def event(self, kind: str, msg: str, **kw):
        t = self.io.now() - (self.t0_sim or self.io.now())
        rec = {'t_sim': round(t, 1), 'wall': time.time(), 'kind': kind, 'msg': msg, **kw}
        self.events.append(rec)
        self.log.info(f'[{kind}] {msg}')
        self.event_pub.publish(String(data=json.dumps(rec)))
        with open(os.path.join(self.run_dir, 'events.jsonl'), 'a') as f:
            f.write(json.dumps(rec) + '\n')

    def _on_water(self, level: float):
        self.water_log.append((self.io.now() - (self.t0_sim or self.io.now()), level))
        self.water_rose.set()

    def ask(self, image, prompt, schema, think, max_tokens, pid, meta=None):
        """VLM call with the sim paused (drone is hovering on a reached goal)."""
        paused = self.io.pause(True)
        t = time.time()
        try:
            out, tm = self.vlm.ask(image, prompt, schema, think=think, max_tokens=max_tokens,
                                   prompt_id=pid, meta=meta)
            if think and 'error' in out:
                # thinking ran out of tokens: the plan's fallback is the same call without thinking
                self.event('vlm', f'{pid}: thinking failed ({out["error"][:60]}), retrying with thinking off')
                out, tm2 = self.vlm.ask(image, prompt, schema, think=False, max_tokens=max_tokens,
                                        prompt_id=pid + '_nothink', meta=meta)
                tm = dict(tm2, wall_s=tm['wall_s'] + tm2['wall_s'], think_fallback=True)
        finally:
            if paused:
                self.io.pause(False)
        self.t_vlm += time.time() - t
        self.event('vlm', f'{pid}: {tm["wall_s"]:.1f} s, prompt {tm.get("prompt_n")} tok, '
                          f'gen {tm.get("predicted_n")} tok' + (' ERROR' if 'error' in out else ''),
                   prompt_id=pid, timings=tm)
        return out, tm

    def fly(self, x, y, z) -> bool:
        ts, tw = self.io.now(), time.time()
        ok = self.io.goto_world(x, y, z)
        self.t_flight_sim += self.io.now() - ts
        self.t_flight_wall += time.time() - tw
        self.publish_markers()
        return ok

    def clamp_xy(self, x, y):
        a = self.cfg['area']
        return min(max(x, a['xmin']), a['xmax']), min(max(y, a['ymin']), a['ymax'])

    # ── planning ──────────────────────────────────────────────────────────────
    def make_plan(self, text: str) -> dict:
        a, h = self.cfg['area'], self.cfg['home']
        prompt = prompts.P_PLAN.format(mission=text, hx=h['x'], hy=h['y'], **a)
        mt = self.cfg['llm']['max_tokens_think']
        think = self.args.get('think_plan', False)
        out, _ = self.ask(None, prompt, schemas.PLAN, think, mt, 'plan')
        ok, errs = validate(out, self.cfg) if 'error' not in out else (False, [out['error']])
        info = {'raw_first': out, 'valid_first_try': ok, 'errors_first': errs}
        if not ok:
            self.event('plan', f'plan invalid: {errs}; asking once more')
            out, _ = self.ask(None, prompt + '\n\n' + prompts.P_PLAN_FIX.format(errors='; '.join(errs)),
                              schemas.PLAN, think, mt, 'plan_fix')
            ok, errs2 = validate(out, self.cfg) if 'error' not in out else (False, [out['error']])
            info.update({'raw_second': out, 'valid_second_try': ok, 'errors_second': errs2})
        if ok:
            plan = normalize(out, self.cfg)
            info['source'] = 'vlm'
            if plan != out:
                info['normalized'] = True
                self.event('plan', 'plan completed with missing triage/beacon/report steps')
        else:
            plan = default_plan(self.cfg)
            info['source'] = 'default'
            self.event('plan', 'plan still invalid: using the default plan')
        info['plan'] = plan
        self.plan_info = info
        self.event('plan', ' -> '.join(f"{s['tool']}{json.dumps(s['args']) if s['args'] else ''}"
                                        for s in plan['steps']), valid_first_try=info['valid_first_try'])
        return plan

    # ── tools ─────────────────────────────────────────────────────────────────
    def t_takeoff(self, alt: float = 20.0, **_):
        h = self.cfg['home']
        self.fly(h['x'], h['y'], float(alt))

    def t_survey(self, alt: float = 20.0, cells=None, **_):
        from flood_vlm.validator import survey_cells
        cells = cells or survey_cells(self.cfg)
        for k, (x, y) in enumerate(cells):
            self.safe_point()
            self.fly(x, y, float(alt))
            frame = self.io.grab()
            if frame is None:
                self.event('survey', f'cell {k}: no image')
                continue
            out, _ = self.ask(frame.image, prompts.P_SURVEY, schemas.SURVEY, False,
                              self.cfg['llm'].get('max_tokens_survey', 768), f'survey{k}',
                              meta={'cell': [x, y], 'cam': frame.t.tolist()})
            if 'error' in out:
                continue
            dets = process_survey(out, frame, self.mem, f'survey{k}')
            self.event('survey', f'cell {k} ({x:.0f}, {y:.0f}): {len(dets)} people, '
                                 f'{len(out.get("dry_areas", []))} dry areas, '
                                 f'flooded {out.get("flooded_fraction")}')
            self.publish_markers()
        self.event('survey', f'memory: {len(self.mem.active())} people '
                             f'{[p.id for p in self.mem.active()]}')
        if self.args.get('lookagain', True):
            for p in sorted(self.mem.active(), key=lambda q: q.conf):
                if p.conf < self.cfg['lookagain_conf'] or p.posture == 'unsure' or p.beacon_xy is None:
                    self.t_look_again(p.id)

    def t_look_again(self, person_id: str, **_):
        p = self.mem.get(person_id)
        if p is None or p.status == 'rejected':
            return
        self.safe_point()
        x, y, z = look_position(p, self.cfg['look_alt'], LOOK_MIN_ABOVE_TARGET_M)
        x, y = self.clamp_xy(x, y)
        self.fly(x, y, z)
        frame = self.io.grab()
        if frame is None:
            return
        out, _ = self.ask(frame.image, look_prompt(p, frame), schemas.LOOK, False,
                          self.cfg['llm']['max_tokens_fast'], f'look_{p.id}',
                          meta={'person': p.id, 'cam': frame.t.tolist()})
        if 'error' in out:
            return
        res = process_look(out, frame, p, self.mem, f'look{p.n_obs}')
        if res.get('person_present') and 'xyz' not in res:
            self.event('look', f'{p.id}: person reported but no usable box: {out.get("bbox_2d")}')
            return
        if not res.get('person_present'):
            if p.looked or p.conf < self.cfg['lookagain_conf']:
                p.status = 'rejected'
                self.event('look', f'{p.id}: no person at close range -> rejected')
            else:
                p.conf *= 0.5
                self.event('look', f'{p.id}: not confirmed, confidence halved')
            return
        # the closer look can show two people merged at survey range: drop duplicates
        for q in self.mem.active():
            if q is not p and math.dist(q.xyz_world[:2], p.xyz_world[:2]) < self.cfg['merge_radius_m']:
                q.status = 'rejected'
        self.event('look', f'{p.id}: {p.posture}, {p.location} at ({p.xyz_world[0]:.1f}, '
                           f'{p.xyz_world[1]:.1f}); beacon {p.beacon_xy and [round(v, 1) for v in p.beacon_xy]} '
                           f'({p.beacon_reason})')
        self.publish_markers()

    def t_triage(self, **_):
        people = self.mem.active()
        if not people:
            self.ranking = []
            return
        w = self.io.water
        prompt = prompts.P_TRIAGE.format(water=w, people=self.mem.to_text(w))
        out, _ = self.ask(None, prompt, schemas.TRIAGE, self.args.get('think_triage', False),
                          self.cfg['llm']['max_tokens_think'], 'triage', meta={'water': w})
        ranking = []
        seen = set()
        for r in (out.get('ranking', []) if 'error' not in out else []):
            pid = r.get('person_id')
            if self.mem.get(pid) is None or pid in seen or self.mem.get(pid).status == 'rejected':
                continue
            seen.add(pid)
            ranking.append({'person_id': pid, 'level': r.get('level', 'medium'),
                            'reason': r.get('reason', '')})
        for p in people:                                 # anyone the VLM forgot goes last
            if p.id not in seen:
                ranking.append({'person_id': p.id, 'level': 'medium', 'reason': 'not ranked by triage'})
        for r in ranking:
            p = self.mem.get(r['person_id'])
            p.level, p.reason = r['level'], r['reason']
        self.ranking = ranking
        self.ranking_water = w
        self.water_rose.clear()
        self.event('triage', f'water {w:.1f} m: ' + ', '.join(f"{r['person_id']}={r['level']}"
                                                             for r in ranking))
        self.publish_markers()

    def t_mark_beacon(self, person_id: str = 'all', **_):
        if person_id != 'all':
            self._beacon(person_id)
            return
        if not self.ranking:
            self.t_triage()
        done = {b['person_id'] for b in self.beacons}
        while True:
            self.safe_point()
            todo = [r['person_id'] for r in self.ranking if r['person_id'] not in done
                    and self.mem.get(r['person_id']).status != 'rejected']
            if not todo:
                break
            self._beacon(todo[0])
            done.add(todo[0])

    def _beacon(self, pid: str):
        p = self.mem.get(pid)
        if p is None or p.status == 'rejected':
            return
        if p.beacon_xy is None or p.surface_dry is False:
            self.event('beacon', f'{pid}: no dry beacon point yet, looking again')
            p.beacon_xy = None
            self.t_look_again(pid)
            if p.status == 'rejected':
                return
        bx, by, bz, why = self.beacon_spot(p)
        if why:
            p.beacon_reason = (p.beacon_reason + ' | ' if p.beacon_reason else '') + why
        self.fly(bx, by, bz + BEACON_HOVER_ABOVE_M)
        self.io.sleep_sim(BEACON_HOVER_S)
        ok = gz_tools.spawn_beacon(pid, bx, by, bz, self.cfg['world_name'])
        level = p.level or 'medium'
        b = {'person_id': pid, 'x': bx, 'y': by, 'z': bz, 'water': self.io.water, 'level': level,
             'spawned': ok, 't_sim': round(self.io.now() - self.t0_sim, 1)}
        self.beacons.append(b)
        p.status = 'beacon_placed'
        req = {'person_id': pid, 'world_xy': [round(v, 2) for v in p.xyz_world[:2]], 'level': level,
               'posture': p.posture, 'location': p.location, 'beacon_xy': [round(bx, 2), round(by, 2)],
               'reason': p.reason or p.beacon_reason, 't': b['t_sim']}
        self.req_pub.publish(String(data=json.dumps(req)))
        self.event('beacon', f'{pid} ({level}): beacon at ({bx:.1f}, {by:.1f}, {bz:.1f})'
                             + ('' if ok else ' - SPAWN FAILED'), request=req)
        self.publish_markers()

    def beacon_spot(self, p) -> tuple[float, float, float, str]:
        """Beacon position: the VLM's point if it is 1.5-15 m from the person; otherwise a
        point BEACON_PUSH_M away, preferably inside a dry area the survey saw (the person's
        own roof, or for people in the water the nearest mound / raised road)."""
        px, py = p.xyz_world[:2]
        dmin = self.cfg['beacon_min_dist_from_person_m']
        if p.beacon_xy is not None and dmin + 0.3 <= math.dist(p.beacon_xy, (px, py)) <= 15.0:
            z = p.beacon_z if p.beacon_z is not None else p.xyz_world[2]
            return p.beacon_xy[0], p.beacon_xy[1], z, ''
        roof = p.location == 'on_roof'
        zones = [zn for zn in self.mem.zones if (zn.kind == 'roof') == roof]
        if roof:
            zones = [zn for zn in zones if G.point_in_polygon(px, py, zn.polygon)]
        cands = []
        for k in range(24):
            a = 2 * math.pi * k / 24
            for r in ((BEACON_PUSH_M,) if roof else (BEACON_PUSH_M, 4.0, 6.0, 9.0, 12.0)):
                cands.append((px + r * math.cos(a), py + r * math.sin(a), r))
        inside = [c for c in cands if any(G.point_in_polygon(c[0], c[1], zn.polygon) for zn in zones)]
        if inside:
            ref = p.beacon_xy or [self.cfg['home']['x'], self.cfg['home']['y']]
            x, y, r = min(inside, key=lambda c: (c[2], math.dist(c[:2], ref)))
            zn = next(zn for zn in zones if G.point_in_polygon(x, y, zn.polygon))
            z = p.xyz_world[2] if roof else zn.z
            return x, y, z, f'moved to {r:.1f} m inside surveyed dry {zn.kind} {zn.id}'
        # no dry area known: keep the VLM's direction (or face home), BEACON_PUSH_M out
        ref = p.beacon_xy if p.beacon_xy and math.dist(p.beacon_xy, (px, py)) > 0.2 else \
            [self.cfg['home']['x'], self.cfg['home']['y']]
        dx, dy = ref[0] - px, ref[1] - py
        n = math.hypot(dx, dy) or 1.0
        z = p.beacon_z if p.beacon_z is not None else p.xyz_world[2]
        return px + BEACON_PUSH_M * dx / n, py + BEACON_PUSH_M * dy / n, z, \
            f'pushed to {BEACON_PUSH_M} m (no surveyed dry area)'

    def t_report(self, **_):
        self.write_report()

    def t_return_home(self, **_):
        h = self.cfg['home']
        self.fly(h['x'], h['y'], self.cfg['survey_alt'])
        pad = self.io.ow.offset[2]                 # odom z = 0 is the pad
        self.fly(h['x'], h['y'], pad + 0.15)
        self.event('home', 'landed at home')

    # ── water rise -> re-plan ─────────────────────────────────────────────────
    def safe_point(self):
        """Between tools: if the water rose since the last triage, re-rank the rest."""
        if not self.water_rose.is_set():
            return
        self.water_rose.clear()
        w = self.io.water
        if not self.ranking:
            self.event('water', f'water rose to {w:.1f} m (no ranking yet)')
            return
        old = [r['person_id'] for r in self.ranking]
        old_levels = {r['person_id']: r['level'] for r in self.ranking}
        old_w = self.ranking_water
        self.t_triage()
        new = [r['person_id'] for r in self.ranking]
        done = {b['person_id'] for b in self.beacons}
        rest_old = [x for x in old if x not in done]
        rest_new = [x for x in new if x not in done]
        moved = [x for x in rest_new if x in rest_old and rest_new.index(x) < rest_old.index(x)]
        why = []
        for x in moved:
            p = self.mem.get(x)
            margin = p.xyz_world[2] - w if p.location == 'on_roof' else 0.0
            why.append(f'{x} moved up: {p.level}, roof margin {margin:.1f} m')
        self.replans.append({'t_sim': round(self.io.now() - self.t0_sim, 1), 'old_water': old_w,
                             'new_water': w, 'old_order': old, 'new_order': new, 'moved_up': moved,
                             'old_levels': old_levels,
                             'new_levels': {r['person_id']: r['level'] for r in self.ranking},
                             'done': sorted(done)})
        self.event('replan', f're-plan because water rose to {w:.1f} m; '
                             + ('; '.join(why) if why else 'order unchanged'))

    # ── markers ───────────────────────────────────────────────────────────────
    def publish_markers(self):
        arr = MarkerArray()
        stamp = self.io.get_clock().now().to_msg()

        def mk(ns, i, mtype, xyz, scale, rgb, text=''):
            m = Marker()
            m.header.frame_id = 'odom'
            m.header.stamp = stamp
            m.ns, m.id, m.type, m.action = ns, i, mtype, Marker.ADD
            o = self.io.ow.world_to_odom(xyz)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, o)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = float(scale)
            m.color.r, m.color.g, m.color.b = map(float, rgb)
            m.color.a = 1.0
            m.text = text
            return m

        for i, p in enumerate(self.mem.people):
            if p.status == 'rejected':
                continue
            arr.markers.append(mk('people', i, Marker.SPHERE, p.xyz_world, 1.0, LEVEL_RGB.get(p.level)))
            arr.markers.append(mk('labels', i, Marker.TEXT_VIEW_FACING,
                                  [p.xyz_world[0], p.xyz_world[1], p.xyz_world[2] + 1.5], 1.0,
                                  (1, 1, 1), f'{p.id} {p.level or ""}'))
        for i, b in enumerate(self.beacons):
            arr.markers.append(mk('beacons', i, Marker.CYLINDER, [b['x'], b['y'], b['z'] + 0.6], 0.6,
                                  (1.0, 0.45, 0.0)))
        if len(self.io.path_world) > 1:
            m = mk('path', 0, Marker.LINE_STRIP, (0, 0, 0), 0.2, (0.1, 0.9, 0.3))
            m.pose.position.x = m.pose.position.y = m.pose.position.z = 0.0
            from geometry_msgs.msg import Point
            for _, x, y, z in self.io.path_world[-500:]:
                o = self.io.ow.world_to_odom((x, y, z))
                m.points.append(Point(x=float(o[0]), y=float(o[1]), z=float(o[2])))
            arr.markers.append(m)
        self.marker_pub.publish(arr)

    # ── run ───────────────────────────────────────────────────────────────────
    def run(self, text: str):
        self.t0_sim = self.io.now()
        self.t0_wall = time.time()
        self.start_pub.publish(Empty())
        self.event('start', f'mission: "{text}"', run_id=self.run_id, seed=self.args['seed'],
                   lookagain=self.args.get('lookagain', True))
        plan = self.make_plan(text)
        tools = {'takeoff': self.t_takeoff, 'survey': self.t_survey, 'look_again': self.t_look_again,
                 'triage': self.t_triage, 'mark_beacon': self.t_mark_beacon, 'report': self.t_report,
                 'return_home': self.t_return_home}
        reported = False
        for s in plan['steps']:
            self.safe_point()
            self.event('tool', f"{s['tool']} {json.dumps(s['args'])}")
            try:
                tools[s['tool']](**s['args'])
            except TypeError as e:
                self.event('tool', f"{s['tool']} bad args {s['args']}: {e}; using defaults")
                tools[s['tool']]()
            reported |= s['tool'] == 'report'
        self.t_total_wall = time.time() - self.t0_wall
        self.write_report()           # final version (incl. return home + timings)
        self.event('done', f'mission finished in {self.t_total_wall:.0f} s wall; report in {self.run_dir}')

    def write_report(self):
        gt_path = os.path.join(drone_worlds_dir(), f'flood_town_{self.args["seed"]}.gt.json')
        gt = G.load_gt(gt_path) if os.path.exists(gt_path) else None
        R.write_all(self, gt)


class MissionNode(Node):
    """Parameter holder so the mission is configurable with --ros-args."""

    def __init__(self):
        super().__init__('mission_params')
        self.declare_parameter('seed', 7)
        self.declare_parameter('mission', DEFAULT_MISSION)
        self.declare_parameter('lookagain', True)
        self.declare_parameter('think_plan', False)     # thinking needs > 3k tokens on CPU (NOTES)
        self.declare_parameter('think_triage', False)
        self.declare_parameter('runs_root', 'runs')
        self.declare_parameter('run_id', '')
        self.declare_parameter('image_long_side', 0)
        self.declare_parameter('config', '')


def main():
    rclpy.init()
    pn = MissionNode()
    gp = lambda n: pn.get_parameter(n).value  # noqa: E731
    cfg = load_config(gp('config') or None)
    args = {'seed': int(gp('seed')), 'lookagain': bool(gp('lookagain')),
            'think_plan': bool(gp('think_plan')), 'think_triage': bool(gp('think_triage')),
            'runs_root': gp('runs_root'),
            'run_id': gp('run_id') or None, 'image_long_side': int(gp('image_long_side')) or None}
    text = gp('mission')
    pn.destroy_node()
    io = DroneIO(cfg, 'flood_mission')
    io.start()
    log = io.get_logger()
    vlm_ok = VLMClient(cfg['llm']['url']).healthy()
    if not vlm_ok:
        log.error('llama-server is not answering on ' + cfg['llm']['url'] + ' (scripts/run_vlm.sh)')
    elif not io.wait_ready():
        log.error('no odometry/ground truth/camera_info - is the flood world running?')
    else:
        Mission(io, cfg, args).run(text)
    io.stop()
    io.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
