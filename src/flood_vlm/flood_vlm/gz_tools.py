"""Thin wrappers over `gz service` (FLOOD_VLM_PLAN.md 1.4).

Each call runs the gz CLI with a 3 s timeout and returns True on success.
"""
from __future__ import annotations

import math
import subprocess

WORLD = 'flood_town'
TIMEOUT_MS = 3000


def _service(world: str, name: str, reqtype: str, req: str,
             reptype: str = 'gz.msgs.Boolean', timeout_ms: int = TIMEOUT_MS) -> bool:
    cmd = ['gz', 'service', '-s', f'/world/{world}/{name}', '--reqtype', reqtype,
           '--reptype', reptype, '--timeout', str(timeout_ms), '--req', req]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_ms / 1000 + 2)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return out.returncode == 0 and 'data: true' in out.stdout


def pause(paused: bool, world: str = WORLD) -> bool:
    return _service(world, 'control', 'gz.msgs.WorldControl',
                    f'pause: {"true" if paused else "false"}')


def set_pose(name: str, x: float, y: float, z: float, yaw: float = 0.0,
             world: str = WORLD) -> bool:
    qz, qw = math.sin(yaw / 2), math.cos(yaw / 2)
    req = (f'name: "{name}", position: {{x: {x}, y: {y}, z: {z}}}, '
           f'orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}')
    return _service(world, 'set_pose', 'gz.msgs.Pose', req)


WATER_HALF_THICKNESS = 0.025   # flood_water is an 80 x 80 x 0.05 box


def set_water(z: float, world: str = WORLD) -> bool:
    """Put the top of the flood_water slab at height z."""
    return set_pose('flood_water', 0.0, 0.0, z - WATER_HALF_THICKNESS, world=world)


BEACON_SDF = """<?xml version="1.0"?>
<sdf version="1.8">
  <model name="beacon_{id}">
    <static>true</static>
    <link name="link">
      <visual name="pole">
        <pose>0 0 0.6 0 0 0</pose>
        <cast_shadows>false</cast_shadows>
        <geometry><cylinder><radius>0.3</radius><length>1.2</length></cylinder></geometry>
        <material><ambient>1 0.45 0 1</ambient><diffuse>1 0.45 0 1</diffuse>
          <emissive>0.6 0.25 0 1</emissive></material>
      </visual>
      <visual name="lamp">
        <pose>0 0 1.45 0 0 0</pose>
        <cast_shadows>false</cast_shadows>
        <geometry><sphere><radius>0.3</radius></sphere></geometry>
        <material><ambient>1 0.9 0.2 1</ambient><diffuse>1 0.9 0.2 1</diffuse>
          <emissive>1 0.85 0.1 1</emissive></material>
      </visual>
    </link>
  </model>
</sdf>
"""


def spawn_beacon(bid: str, x: float, y: float, z: float, world: str = WORLD) -> bool:
    """Orange cylinder (r 0.3, h 1.2) + glowing sphere, visual only, named beacon_<id>.

    The SDF goes inline: `create` is asynchronous, so a temp `sdf_filename` can be
    deleted before the server reads it.
    """
    sdf = ' '.join(BEACON_SDF.format(id=bid).replace('"', "'").split())
    req = (f'sdf: "{sdf}", name: "beacon_{bid}", allow_renaming: false, '
           f'pose: {{position: {{x: {x}, y: {y}, z: {z}}}}}')
    return _service(world, 'create', 'gz.msgs.EntityFactory', req)


def remove(name: str, world: str = WORLD) -> bool:
    return _service(world, 'remove', 'gz.msgs.Entity', f'name: "{name}", type: MODEL')
