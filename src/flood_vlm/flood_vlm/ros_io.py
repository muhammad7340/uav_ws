"""ROS side of the drone for blocking mission code.

DroneIO spins its own executor in a background thread, so the caller (mission
node, try_survey.py, data collection) can write straight-line code:

    io = DroneIO(cfg); io.start(); io.wait_ready()
    io.goto_world(0, -26, 20)
    frame = io.grab()          # image + K + camera pose in WORLD frame
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32, String
from tf2_ros import Buffer, TransformListener

from flood_vlm import geometry as geo
from flood_vlm import gz_tools

CAMERA_FRAME = 'camera_down_link_optical'
ODOM_FRAME = 'odom'


@dataclass
class Frame:
    image: np.ndarray            # BGR
    K: np.ndarray
    R: np.ndarray                # optical -> world
    t: np.ndarray                # camera position, world
    stamp: float                 # sim seconds
    drone_world: np.ndarray      # ground truth (for scoring only)
    pose_source: str             # 'tf' or 'gt'
    water: float

    @property
    def size(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]


class DroneIO(Node):
    def __init__(self, cfg: dict, name: str = 'flood_drone_io'):
        super().__init__(name, parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.cfg = cfg
        self.world_name = cfg['world_name']
        self.bridge = CvBridge()
        grp = ReentrantCallbackGroup()
        self._lock = threading.Lock()
        self.odom = None            # latest EKF position (odom)
        self.gt = None              # latest ground-truth position (world)
        self.gt_quat = None
        self.state = ''
        self.reached = False
        self._reached_seq = 0
        self.image_msg = None
        self.K = None
        self.water = float(cfg['water']['start'])
        self._water_seq = 0
        self.ow: geo.OdomWorld | None = None
        self.path_world: list[tuple[float, float, float, float]] = []   # (t, x, y, z) GT path

        self.create_subscription(Odometry, 'odometry/filtered', self._odom_cb, 10, callback_group=grp)
        self.create_subscription(Odometry, 'drone/odom_ground_truth', self._gt_cb, 10, callback_group=grp)
        self.create_subscription(String, 'drone/state', self._state_cb, 10, callback_group=grp)
        self.create_subscription(Bool, 'drone/goal_reached', self._reached_cb, 10, callback_group=grp)
        self.create_subscription(Image, '/camera_down/image', self._img_cb, qos_profile_sensor_data,
                                 callback_group=grp)
        self.create_subscription(CameraInfo, '/camera_down/camera_info', self._info_cb,
                                 qos_profile_sensor_data, callback_group=grp)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Float32, '/flood/water_level', self._water_cb, latched, callback_group=grp)
        self.goal_pub = self.create_publisher(PoseStamped, 'drone/goal_pose', 10)
        self.tf_buffer = Buffer(cache_time=Duration(seconds=30))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self._executor = None
        self._thread = None
        self.water_listeners = []       # callables(level), called from the executor thread

    # ── callbacks ─────────────────────────────────────────────────────────────
    def _odom_cb(self, m):
        p = m.pose.pose.position
        self.odom = np.array([p.x, p.y, p.z])

    def _gt_cb(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.gt = np.array([p.x, p.y, p.z])
        self.gt_quat = (q.x, q.y, q.z, q.w)
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if not self.path_world or t - self.path_world[-1][0] > 0.5:
            self.path_world.append((t, p.x, p.y, p.z))

    def _state_cb(self, m):
        self.state = m.data

    def _reached_cb(self, m):
        with self._lock:
            self.reached = m.data
            if m.data:
                self._reached_seq += 1

    def _img_cb(self, m):
        self.image_msg = m

    def _info_cb(self, m):
        if self.K is None:
            self.K = np.array(m.k, float).reshape(3, 3)

    def _water_cb(self, m):
        changed = abs(m.data - self.water) > 1e-3
        self.water = float(m.data)
        self._water_seq += 1
        if changed:
            for f in self.water_listeners:
                f(self.water)

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self):
        self._executor = MultiThreadedExecutor(num_threads=3)
        self._executor.add_node(self)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def stop(self):
        if self._executor:
            self._executor.shutdown()

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def sleep_sim(self, seconds: float, wall_timeout: float | None = None):
        t_end = self.now() + seconds
        w_end = time.time() + (wall_timeout or max(30.0, seconds * 20))
        while self.now() < t_end and time.time() < w_end:
            time.sleep(0.05)

    def wait_ready(self, timeout: float = 180.0) -> bool:
        """Odometry, ground truth and camera_info received; measure the odom->world offset."""
        end = time.time() + timeout
        while time.time() < end:
            if self.odom is not None and self.gt is not None and self.K is not None and self.now() > 0:
                break
            time.sleep(0.2)
        else:
            return False
        # average a few samples while sitting on the pad
        samples = []
        for _ in range(10):
            samples.append(self.gt - self.odom)
            time.sleep(0.1)
        self.ow = geo.OdomWorld(np.mean(samples, axis=0))
        self.get_logger().info(f'odom->world offset {np.round(self.ow.offset, 3).tolist()}')
        return True

    # ── flight ────────────────────────────────────────────────────────────────
    def drone_world(self) -> np.ndarray:
        """Drone position in world frame from the EKF (what the agent is allowed to know)."""
        return self.ow.odom_to_world(self.odom)

    def send_goal_world(self, x: float, y: float, z: float):
        o = self.ow.world_to_odom((x, y, z))
        msg = PoseStamped()
        msg.header.frame_id = ODOM_FRAME
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, o)
        msg.pose.orientation.w = 1.0
        with self._lock:
            self.reached = False
            seq = self._reached_seq
        self.goal_pub.publish(msg)
        return seq

    def goto_world(self, x: float, y: float, z: float, timeout_s: float | None = None) -> bool:
        """Fly to a world pose and block until drone/goal_reached (sim-time timeout)."""
        cur = self.drone_world()
        d_xy = math.hypot(x - cur[0], y - cur[1])
        d_z = abs(z - cur[2])
        if timeout_s is None:
            timeout_s = 2.5 * (d_xy / self.cfg['cruise_speed'] + d_z / self.cfg['descend_speed']) + 30
        seq = self.send_goal_world(x, y, z)
        t_end = self.now() + timeout_s
        w_end = time.time() + timeout_s * 30
        while self.now() < t_end and time.time() < w_end:
            with self._lock:
                if self._reached_seq > seq:
                    return True
            time.sleep(0.05)
        self.get_logger().warn(f'goal ({x:.1f}, {y:.1f}, {z:.1f}) not reached in {timeout_s:.0f} sim-s')
        return False

    # ── perception ────────────────────────────────────────────────────────────
    def grab(self, settle_s: float | None = None, timeout: float = 60.0) -> Frame | None:
        """Wait settle_s (sim), then return the first image newer than that with its pose."""
        settle_s = self.cfg['settle_s'] if settle_s is None else settle_s
        self.sleep_sim(settle_s)
        t_min = self.now()
        end = time.time() + timeout
        msg = None
        while time.time() < end:
            m = self.image_msg
            if m is not None and m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 >= t_min:
                msg = m
                break
            time.sleep(0.05)
        if msg is None:
            return None
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        R, t, src = self.camera_pose_world(msg.header.stamp)
        return Frame(image=img, K=self.K.copy(), R=R, t=t, stamp=stamp,
                     drone_world=self.gt.copy(), pose_source=src, water=self.water)

    def camera_pose_world(self, stamp) -> tuple[np.ndarray, np.ndarray, str]:
        """odom -> camera optical from TF (EKF), shifted to world. Falls back to ground truth."""
        try:
            tf = self.tf_buffer.lookup_transform(ODOM_FRAME, CAMERA_FRAME, Time.from_msg(stamp),
                                                 timeout=Duration(seconds=1.0))
            q = tf.transform.rotation
            p = tf.transform.translation
            R = geo.quat_to_matrix(q.x, q.y, q.z, q.w)
            t = self.ow.odom_to_world((p.x, p.y, p.z))
            return R, t, 'tf'
        except Exception as e:  # noqa: BLE001 - tf2 raises several types
            self.get_logger().warn(f'TF lookup failed ({e}); using ground-truth pose')
            R, t = geo.camera_pose_from_base(self.gt, self.gt_quat)
            return R, t, 'gt'

    # ── sim control ───────────────────────────────────────────────────────────
    def pause(self, paused: bool) -> bool:
        return gz_tools.pause(paused, self.world_name)
