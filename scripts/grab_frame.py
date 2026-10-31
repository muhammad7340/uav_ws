#!/usr/bin/env python3
"""Save one frame from a ROS image topic to a PNG.

    python3 scripts/grab_frame.py /camera_down/image docs/flood/down_20m.png
    python3 scripts/grab_frame.py /flood/overview/image docs/flood/scene_seed7.png --timeout 120
"""
import argparse
import sys

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('topic')
    ap.add_argument('out')
    ap.add_argument('--timeout', type=float, default=30.0)
    ap.add_argument('--skip', type=int, default=0, help='frames to drop first (stale buffers)')
    args = ap.parse_args()

    rclpy.init()
    node = Node('grab_frame')
    bridge = CvBridge()
    state = {'n': 0, 'done': False}

    def cb(msg):
        state['n'] += 1
        if state['n'] <= args.skip or state['done']:
            return
        img = bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv2.imwrite(args.out, img)
        node.get_logger().info(f'saved {args.out} ({msg.width}x{msg.height})')
        state['done'] = True

    node.create_subscription(Image, args.topic, cb, qos_profile_sensor_data)
    end = node.get_clock().now().nanoseconds + args.timeout * 1e9
    while rclpy.ok() and not state['done'] and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    ok = state['done']
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
