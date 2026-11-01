#!/usr/bin/env python3
"""Raise the flood water on a sim-time schedule (FLOOD_VLM_PLAN.md 1.4).

Publishes /flood/water_level (std_msgs/Float32, latched) and moves the
`flood_water` slab, the floating debris and the rescue boat with it.
Publish a Float32 on /flood/set_water_level to force a level (tests, eval).
"""
from __future__ import annotations

import json
import math
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, Float32

from flood_vlm import gz_tools
from flood_vlm.config import load_config

DEBRIS_BELOW_SURFACE = 0.2   # log origin sits this far below the water line


class WaterController(Node):
    def __init__(self):
        super().__init__('water_controller')
        cfg = load_config()
        self.declare_parameter('world', cfg['world_name'])
        self.declare_parameter('start', float(cfg['water']['start']))
        self.declare_parameter('steps', [float(s) for s in cfg['water']['steps']])
        self.declare_parameter('step_every_s', float(cfg['water']['step_every_s']))
        self.declare_parameter('enabled', True)        # false: only manual levels
        self.declare_parameter('gt_path', '')          # *.gt.json, to move debris + boat
        self.declare_parameter('start_on_mission', False)  # clock starts on /flood/mission_started

        self.world = self.get_parameter('world').value
        self.level = float(self.get_parameter('start').value)
        self.steps = list(self.get_parameter('steps').value)
        self.every = float(self.get_parameter('step_every_s').value)
        self.enabled = bool(self.get_parameter('enabled').value)
        self.floaters = self._load_floaters(self.get_parameter('gt_path').value)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(Float32, '/flood/water_level', latched)
        self.create_subscription(Float32, '/flood/set_water_level', self._manual_cb, 10)
        self.t0 = None
        self.wait_mission = bool(self.get_parameter('start_on_mission').value)
        self.create_subscription(Empty, '/flood/mission_started', self._mission_cb, 10)
        self.next_idx = 0
        self._lock = threading.Lock()
        self.create_timer(0.5, self._tick)
        self.pub.publish(Float32(data=self.level))
        self.get_logger().info(
            f'water at {self.level:.2f} m; steps {self.steps} every {self.every:.0f} sim-s'
            + ('' if self.enabled else ' (schedule disabled)'))

    def _load_floaters(self, path: str) -> list[dict]:
        if not path:
            return []
        try:
            with open(path) as f:
                gt = json.load(f)
        except OSError as e:
            self.get_logger().warn(f'cannot read {path}: {e}')
            return []
        out = [{'name': d['id'], 'x': d['x'], 'y': d['y'], 'yaw': d.get('yaw', 0.0),
                'dz': -DEBRIS_BELOW_SURFACE}
               for d in gt.get('debris', [])]
        h = gt.get('home', {})
        if h:
            out.append({'name': 'rc_boat', 'x': h['x'] + 4.0, 'y': h['y'] + 4.6,
                        'yaw': math.pi / 2, 'dz': 0.0})
        return out

    def _tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now <= 0.0:                 # no /clock yet
            return
        if self.t0 is None:
            if self.wait_mission:
                return
            self.t0 = now
        if not self.enabled or self.next_idx >= len(self.steps):
            return
        if now - self.t0 >= self.every * (self.next_idx + 1):
            level = float(self.steps[self.next_idx])
            self.next_idx += 1
            self.set_level(level, f'schedule step {self.next_idx} at sim t={now - self.t0:.0f} s')

    def _mission_cb(self, _msg: Empty):
        if self.wait_mission and self.t0 is None:
            self.t0 = self.get_clock().now().nanoseconds * 1e-9
            self.get_logger().info('mission started: water clock running')

    def _manual_cb(self, msg: Float32):
        self.set_level(float(msg.data), 'manual')

    def set_level(self, level: float, why: str):
        with self._lock:
            ok = gz_tools.set_water(level, self.world)
            for f in self.floaters:
                gz_tools.set_pose(f['name'], f['x'], f['y'], level + f['dz'], f['yaw'], self.world)
            self.level = level
            self.pub.publish(Float32(data=level))
        self.get_logger().info(f'WATER ROSE to {level:.2f} m ({why}){"" if ok else " - set_pose FAILED"}')


def main():
    rclpy.init()
    node = WaterController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
