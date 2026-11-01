#!/usr/bin/env python3
"""Record a mission for the demo video and figures (reproducible from the run folder).

    source scripts/flood_env.sh
    scripts/record_showcase.sh 7          # launches the mission + this recorder

Spawns two extra cameras into the running flood world (no change to the world file):
  * showcase: oblique view over the whole town (1280x720, 4 Hz)
  * dronecam: close-up of the drone on its pad (1280x720, 1 Hz, for the "our UAV" photo)
and saves, named by sim time:
  <out>/showcase/<t>.jpg, <out>/down/<t>.jpg, <out>/dronecam/<t>.jpg,
  <out>/path.csv (ground-truth drone path), <out>/cameras.json (poses + intrinsics).
scripts/make_video.py turns this plus runs/<id>/events.jsonl + vlm_calls.jsonl into the video.
"""
import argparse
import json
import math
import os
import subprocess
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from flood_vlm import gz_tools

W, H = 1280, 720


def look_at(eye, target):
    dx, dy, dz = (t - e for t, e in zip(target, eye))
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(-dz, math.hypot(dx, dy))     # +pitch = nose down for a gz camera
    return yaw, pitch


CAMS = {
    # name: (eye, target, hfov, rate)
    'showcase': ((-40.0, -56.0, 40.0), (0.0, -4.0, 0.0), 1.15, 4),
    'dronecam': ((2.3, -28.4, 4.1), (0.0, -26.0, 3.2), 0.75, 1),
}


def camera_sdf(name, eye, target, hfov, rate):
    yaw, pitch = look_at(eye, target)
    return f"""<?xml version="1.0"?><sdf version="1.8"><model name="cam_{name}"><static>true</static>
<pose>{eye[0]} {eye[1]} {eye[2]} 0 {pitch} {yaw}</pose><link name="link">
<sensor name="{name}" type="camera"><camera><horizontal_fov>{hfov}</horizontal_fov>
<image><width>{W}</width><height>{H}</height><format>R8G8B8</format></image>
<clip><near>0.1</near><far>300</far></clip></camera><always_on>1</always_on>
<update_rate>{rate}</update_rate><topic>flood/{name}/image</topic></sensor></link></model></sdf>"""


class Recorder(Node):
    def __init__(self, out):
        super().__init__('showcase_recorder', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.out = out
        self.bridge = CvBridge()
        for d in ('showcase', 'down', 'dronecam'):
            os.makedirs(os.path.join(out, d), exist_ok=True)
        self.path = open(os.path.join(out, 'path.csv'), 'a')
        self.last_path = -1.0
        self.n_drone = 0
        for name, topic in (('showcase', '/flood/showcase/image'), ('down', '/camera_down/image'),
                            ('dronecam', '/flood/dronecam/image')):
            self.create_subscription(Image, topic, lambda m, n=name: self.save(n, m), qos_profile_sensor_data)
        self.create_subscription(Odometry, 'drone/odom_ground_truth', self.odom, 10)

    def save(self, name, msg):
        if name == 'dronecam':
            self.n_drone += 1
            if self.n_drone > 8:          # a few frames on the pad are enough
                return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv2.imwrite(os.path.join(self.out, name, f'{t:010.2f}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 88])

    def odom(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if t - self.last_path >= 0.25:
            p = m.pose.pose.position
            self.path.write(f'{t:.2f},{p.x:.3f},{p.y:.3f},{p.z:.3f}\n')
            self.path.flush()
            self.last_path = t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--world', default='flood_town')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta = {}
    for name, (eye, target, hfov, rate) in CAMS.items():
        sdf = ' '.join(camera_sdf(name, eye, target, hfov, rate).replace('"', "'").split())
        for _ in range(30):
            if gz_tools._service(args.world, 'create', 'gz.msgs.EntityFactory',
                                 f'sdf: "{sdf}", name: "cam_{name}", allow_renaming: false'):
                break
            time.sleep(2)
        yaw, pitch = look_at(eye, target)
        fx = (W / 2) / math.tan(hfov / 2)
        meta[name] = {'eye': eye, 'target': target, 'yaw': yaw, 'pitch': pitch, 'hfov': hfov,
                      'K': [[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], 'size': [W, H]}
    with open(os.path.join(args.out, 'cameras.json'), 'w') as f:
        json.dump(meta, f, indent=1)

    bridge = subprocess.Popen(['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
                               '/flood/showcase/image@sensor_msgs/msg/Image[gz.msgs.Image',
                               '/flood/dronecam/image@sensor_msgs/msg/Image[gz.msgs.Image',
                               '--ros-args', '-p', 'use_sim_time:=true'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rclpy.init()
    node = Recorder(args.out)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        bridge.terminate()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
