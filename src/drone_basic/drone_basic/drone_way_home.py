#!/usr/bin/env python3

from __future__ import annotations
import math, rclpy
from collections import deque
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
import threading
import tf_transformations as tft

class GoHome(Node):
    # ───────── initialization ────────────────────────────────────────────────
    def __init__(self) -> None:
        super().__init__('go_home')
        # Parameters
        self.declare_parameter('odom_topic',       'odometry/filtered')
        self.declare_parameter('cmd_vel_topic',    'drone/cmd_vel')
        self.declare_parameter('waypoint_topic',   'drone/waypoint')
        self.declare_parameter('takeoff_height',   3.0)
        self.declare_parameter('ascend_speed',     0.6)
        self.declare_parameter('cruise_speed',     0.5)
        self.declare_parameter('descend_speed',    0.5)
        self.declare_parameter('position_tolerance', 0.3)
        self.declare_parameter('vertical_tolerance', 0.05)
        self.declare_parameter('landing_height',     0.15)
        self.declare_parameter('control_rate',       20.0)  # Hz
        self.declare_parameter('xy_gain', 0.8)
        # 3D goal input (drone/goal_pose, frame odom)
        self.declare_parameter('goal_topic',        'drone/goal_pose')
        self.declare_parameter('goal_z_tolerance',   0.15)
        self.declare_parameter('z_gain',             1.0)
        self.last_odom = None
        self._last_twist = Twist()
        self._recent_twists: deque[Twist] = deque(maxlen=10)  # our own echoes
        
        # Topics / services
        odom_t = self.get_parameter('odom_topic').value
        cmd_t  = self.get_parameter('cmd_vel_topic').value
        wpt_t  = self.get_parameter('waypoint_topic').value
        self.create_subscription(Odometry, odom_t, self._odom_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, cmd_t, 10)
        self.create_subscription(Twist, cmd_t, self._cmd_watch_cb, 10)
        self.create_service(Trigger, 'go_home', self._srv_home_cb)
        self.create_subscription(String, wpt_t, self._waypoint_cb, 10)
        goal_t = self.get_parameter('goal_topic').value
        self.create_subscription(PoseStamped, goal_t, self._goal_cb, 10)
        self.state_pub = self.create_publisher(String, 'drone/state', 10)
        self.reached_pub = self.create_publisher(Bool, 'drone/goal_reached', 10)
        self.create_timer(0.2, self._publish_state)
        # State
        self.current:  tuple[float, float, float] | None = None
        self.home:     tuple[float, float, float] | None = None
        self.wpts:     list[tuple[float, float, float] | None] = [None] * 5
        self.target:   tuple[float, float, float] | None = None
        self.goal:     tuple[float, float, float] | None = None
        self._goal_z_first = True
        self._goal_reached_sent = False
        self.state: str = 'IDLE'
        # self._last_autonomous_pub = self.get_clock().now()  # only when FSM != IDLE
        rate = self.get_parameter('control_rate').value
        self.create_timer(1.0 / rate, self._loop)
        self.get_logger().info('go_home node up -awaiting odometry…')

    # ───────── subscribers / service handlers ────────────────────────────────
    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self.current = (p.x, p.y, p.z)
        self.last_odom = msg                      # ← store full message
        if self.home is None:
            self.home = self.current
            self.get_logger().info(f'Home set at {self.home}')

    def _cmd_watch_cb(self, msg: Twist) -> None:
        """Abort autonomous flight on any external cmd_vel."""
        # ❶ If we’re idle, nothing to cancel
        if self.state == 'IDLE':
            return  

        # ❷ Is this twist numerically identical to one we just issued?
        #    (compare against a short history: under load the echo can arrive
        #    after the next control tick has already published a new twist)
        def same(t: Twist) -> bool:
            return (
                abs(msg.linear.x  - t.linear.x)  < 1e-4 and
                abs(msg.linear.y  - t.linear.y)  < 1e-4 and
                abs(msg.linear.z  - t.linear.z)  < 1e-4 and
                abs(msg.angular.x - t.angular.x) < 1e-4 and
                abs(msg.angular.y - t.angular.y) < 1e-4 and
                abs(msg.angular.z - t.angular.z) < 1e-4
            )
        if any(same(t) for t in self._recent_twists):
            return  # merely our echo

        # ❸ Anything else ⇒ manual override → stop mission
        self.state = 'IDLE'
        self.cmd_pub.publish(Twist())          # full stop
        self.get_logger().warn('cmd_vel override – mission aborted.')

    def _to_body(self, vx_w: float, vy_w: float, yaw: float) -> tuple[float, float]:
        """World-frame → body-frame XY rotation."""
        cos, sin = math.cos(-yaw), math.sin(-yaw)
        return cos * vx_w - sin * vy_w, sin * vx_w + cos * vy_w

    def _srv_home_cb(self, _, res):
        if not self.home or not self.current:
            res.success, res.message = False, 'No odometry yet'
            return res
        if self.state != 'IDLE':
            res.success, res.message = False, f'Busy ({self.state})'
            return res
        self.target = self.home
        self.state = 'HOME_ASCEND'
        self.get_logger().info('Home sequence initiated')
        res.success, res.message = True, 'Going home'
        return res

    # ───────── 3D goal handling ───────────────────────────────────────────────
    def _goal_cb(self, msg: PoseStamped) -> None:
        """New goal pre-empts any current goal (climb first if going up)."""
        if self.current is None:
            self.get_logger().warn('No odometry - cannot fly to goal')
            return
        p = msg.pose.position
        self.goal = (p.x, p.y, p.z)
        self._goal_z_first = p.z > self.current[2]
        self.state = 'GOAL_Z' if self._goal_z_first else 'GOAL_XY'
        self._goal_reached_sent = False
        self.reached_pub.publish(Bool(data=False))
        self.get_logger().info(
            f'Goal ({p.x:.2f}, {p.y:.2f}, {p.z:.2f}) -> {self.state}')

    def _vel_z(self, dz: float, asc: float, dsc: float) -> float:
        """P controller on height, clamped to ascend/descend speed."""
        vz = self.get_parameter('z_gain').value * dz
        return max(-dsc, min(asc, vz))

    def _publish_state(self) -> None:
        self.state_pub.publish(String(data=self.state))

    # ───────── waypoint handling ─────────────────────────────────────────────
    def _vel_xy(self, dx: float, dy: float) -> tuple[float, float]:
        """Simple P controller with speed saturation."""
        k = self.get_parameter('xy_gain').value
        vmax = self.get_parameter('cruise_speed').value
        vx, vy = k * dx, k * dy  # P term
        speed = math.hypot(vx, vy)
        if speed > vmax:  # clamp to cruise_speed
            scale = vmax / speed
            vx *= scale
            vy *= scale
        return vx, vy


    def _waypoint_cb(self, msg: String) -> None:
        key = msg.data.strip()
        # Home keyword over topic ------------------------------------------------
        if key.lower() == 'home':
            if self.state == 'IDLE':
                self.target = self.home
                self.state = 'HOME_ASCEND'
                self.get_logger().info('Home command received')
            else:
                self.get_logger().warn(f'Cannot go home (busy: {self.state})')
            return
        # Waypoints 1‑5 ---------------------------------------------------------
        if key in '12345':
            idx = int(key) - 1
            if self.current is None:
                self.get_logger().warn('No odometry - cannot store/fly waypoint')
                return
            if self.wpts[idx] is None:
                self.wpts[idx] = self.current
                self.get_logger().info(f'Waypoint {key} saved at {self.current}')
            else:
                if self.state != 'IDLE':
                    self.get_logger().warn(f'Busy ({self.state}) ')
                    return
                self.target = self.wpts[idx]
                self.state = 'WP_ASCEND_CRUISE'
                self.get_logger().info(f'Flying to waypoint {key}: {self.target}')
            return
        # Any other key → cancel -------------------------------------------------
        if self.state != 'IDLE':
            self.state = 'IDLE'
            self.get_logger().info('Mission cancelled by user input')

    # ───────── control loop ───────────────────────────────────────────────────
    def _loop(self):
        if self.current is None:
            return
        if self.last_odom is None: 
            return

        x, y, z = self.current
        q = self.last_odom.pose.pose.orientation  
        yaw = tft.euler_from_quaternion(
            [q.x, q.y, q.z, q.w])[2]
        tol_xy = self.get_parameter('position_tolerance').value
        tol_z = self.get_parameter('vertical_tolerance').value
        asc = self.get_parameter('ascend_speed').value
        dsc = self.get_parameter('descend_speed').value
        land_z = self.get_parameter('landing_height').value
        cruise_z = self.get_parameter('takeoff_height').value
        tw = Twist()  # all zeros

        # ── FSM ----------------------------------------------------------------
        if self.state == 'HOME_ASCEND':
            diff = cruise_z - z
            if abs(diff) > tol_z:
                tw.linear.z = asc if diff > 0 else -dsc
            else:
                self.state = 'HOME_CRUISE'; self.get_logger().info('→ HOME_CRUISE')
        elif self.state == 'HOME_CRUISE':
            tx, ty, _ = self.target
            dx, dy = tx - x, ty - y
            dist = math.hypot(dx, dy)
            if dist > tol_xy:
                # proportional XY controller (slows as it nears the goal)
                tw.linear.x, tw.linear.y = self._vel_xy(dx, dy)
            else:
                self.state = 'HOME_DESCEND'
                self.get_logger().info('→ HOME_DESCEND')
        elif self.state == 'HOME_DESCEND':
            if z > land_z + tol_z:
                tw.linear.z = -dsc
            else:
                self.state = 'IDLE'; self.get_logger().info('Home landed')
        elif self.state == 'WP_ASCEND_CRUISE':
            diff = cruise_z - z
            if abs(diff) > tol_z:
                tw.linear.z = asc if diff > 0 else -dsc
            else:
                self.state = 'WP_CRUISE_XY'; self.get_logger().info('WP_CRUISE_XY')
        elif self.state == 'WP_CRUISE_XY':
            tx, ty, _ = self.target
            dx, dy = tx - x, ty - y
            dist = math.hypot(dx, dy)
            if dist > tol_xy:
                tw.linear.x, tw.linear.y = self._vel_xy(dx, dy)
            else:
                self.state = 'WP_ADJUST_Z'
                self.get_logger().info('WP_ADJUST_Z')

        elif self.state == 'WP_ADJUST_Z':
            _, _, tz = self.target
            diff = tz - z
            if abs(diff) > tol_z:
                tw.linear.z = asc if diff > 0 else -dsc
            else:
                self.state = 'IDLE'; self.get_logger().info('Waypoint reached')
        # ── 3D goal: GOAL_Z / GOAL_XY in the order chosen in _goal_cb, then hover
        elif self.state in ('GOAL_Z', 'GOAL_XY', 'GOAL_HOLD'):
            gx, gy, gz = self.goal
            tol_gz = self.get_parameter('goal_z_tolerance').value
            dx, dy, dz = gx - x, gy - y, gz - z
            if self.state == 'GOAL_Z':
                tw.linear.z = self._vel_z(dz, asc, dsc)
                if abs(dz) < tol_gz:
                    self.state = 'GOAL_XY' if self._goal_z_first else 'GOAL_HOLD'
                    self.get_logger().info(f'→ {self.state}')
            elif self.state == 'GOAL_XY':
                tw.linear.x, tw.linear.y = self._vel_xy(dx, dy)
                if self._goal_z_first:          # hold the height reached
                    tw.linear.z = self._vel_z(dz, asc, dsc)
                if math.hypot(dx, dy) < tol_xy:
                    self.state = 'GOAL_HOLD' if self._goal_z_first else 'GOAL_Z'
                    self.get_logger().info(f'→ {self.state}')
            else:  # GOAL_HOLD: hover on the goal
                tw.linear.x, tw.linear.y = self._vel_xy(dx, dy)
                tw.linear.z = self._vel_z(dz, asc, dsc)
            if self.state == 'GOAL_HOLD' and not self._goal_reached_sent:
                self._goal_reached_sent = True  # publish "reached" once per goal
                self.reached_pub.publish(Bool(data=True))
                self.get_logger().info('Goal reached - holding')
        else:  # IDLE – publish NOTHING to avoid masking user cmd_vel
            return

        # Publish and remember time (for cmd_vel filtering) -------------------
        tw.linear.x, tw.linear.y = self._to_body(
        tw.linear.x, tw.linear.y, yaw)
        self._last_twist = tw
        self._recent_twists.append(tw)
        self.cmd_pub.publish(tw)

# ───────── main ───────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = GoHome()

    def spin_node():
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass

    spin_thread = threading.Thread(target=spin_node)
    spin_thread.start()

    try:
        spin_thread.join()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
