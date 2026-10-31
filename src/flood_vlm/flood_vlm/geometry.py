"""Pixel -> world projection for the down camera (FLOOD_VLM_PLAN.md 2.3).

Pure numpy; ROS (camera_info, TF) is only touched by the callers, which pass
K and the camera pose as arrays. Conventions:
  * VLM coordinates are 0-1000 relative (Qwen): u = x / 1000 * W, v = y / 1000 * H.
  * Camera pose = rotation R (optical -> frame) and translation t, optical frame
    is z forward, x right, y down (REP 103).
  * world = odom + offset (EKF odom starts at the spawn point, incl. height).
"""
from __future__ import annotations

import math

import numpy as np


# ───────── rotations ─────────────────────────────────────────────────────────
def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis rpy -> rotation matrix (Rz @ Ry @ Rx)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# Down camera mount from drone_custom.urdf: base_link -> camera_down_link (rpy 0 1.5707 0,
# xyz 0 0 -0.03) -> camera_down_link_optical (rpy -1.5707 0 -1.5707).
R_BASE_CAM = rpy_to_matrix(0.0, 1.5707, 0.0) @ rpy_to_matrix(-1.5707, 0.0, -1.5707)
T_BASE_CAM = np.array([0.0, 0.0, -0.03])


def camera_pose_from_base(base_xyz, base_quat) -> tuple[np.ndarray, np.ndarray]:
    """Optical-frame pose from a base_link pose (used with ground-truth odometry)."""
    rb = quat_to_matrix(*base_quat)
    return rb @ R_BASE_CAM, np.asarray(base_xyz, float) + rb @ T_BASE_CAM


# ───────── intrinsics ────────────────────────────────────────────────────────
def intrinsics_from_hfov(width: int, height: int, hfov: float) -> np.ndarray:
    """K for a Gazebo camera (square pixels, principal point at the centre)."""
    fx = (width / 2.0) / math.tan(hfov / 2.0)
    return np.array([[fx, 0, width / 2.0], [0, fx, height / 2.0], [0, 0, 1.0]])


def rel_to_pixel(x_rel: float, y_rel: float, width: int, height: int) -> tuple[float, float]:
    """0-1000 Qwen coordinates -> pixel coordinates."""
    return x_rel / 1000.0 * width, y_rel / 1000.0 * height


def pixel_to_rel(u: float, v: float, width: int, height: int) -> tuple[float, float]:
    return u / width * 1000.0, v / height * 1000.0


def bbox_center_rel(bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


# ───────── projection ────────────────────────────────────────────────────────
def pixel_ray(u: float, v: float, K: np.ndarray) -> np.ndarray:
    """Direction (optical frame) of the ray through pixel (u, v)."""
    return np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])


def project_to_plane(u: float, v: float, K: np.ndarray, R: np.ndarray, t: np.ndarray,
                     z_plane: float) -> np.ndarray | None:
    """Intersect the ray through (u, v) with the horizontal plane z = z_plane.

    R, t: optical frame -> target frame (the plane height is in that frame).
    Returns xyz, or None if the ray does not hit the plane in front of the camera.
    """
    d = R @ pixel_ray(u, v, K)
    if abs(d[2]) < 1e-9:
        return None
    s = (z_plane - t[2]) / d[2]
    if s <= 0:
        return None
    return t + s * d


def world_to_pixel(p: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[float, float] | None:
    """Inverse projection (for GT overlays and FloodSim-VQA). None if behind the camera."""
    pc = R.T @ (np.asarray(p, float) - t)
    if pc[2] <= 1e-6:
        return None
    return K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]


def ground_footprint(K: np.ndarray, R: np.ndarray, t: np.ndarray, width: int, height: int,
                     z_plane: float) -> list[np.ndarray]:
    """World corners of the image on the plane z = z_plane (for coverage plots)."""
    out = []
    for u, v in ((0, 0), (width, 0), (width, height), (0, height)):
        p = project_to_plane(u, v, K, R, t, z_plane)
        if p is not None:
            out.append(p)
    return out


def pixel_ray_world(u: float, v: float, K: np.ndarray, R: np.ndarray) -> np.ndarray:
    d = R @ pixel_ray(u, v, K)
    return d / np.linalg.norm(d)


def triangulate(rays: list[tuple], min_angle_deg: float = 8.0) -> np.ndarray | None:
    """Least-squares point closest to several rays (origin, unit direction).

    Returns None when the rays are too close to parallel (angle below min_angle_deg),
    which is the case when the survey saw the target almost straight below.
    """
    if len(rays) < 2:
        return None
    dirs = [np.asarray(d, float) for _, d in rays]
    best = max(math.degrees(math.acos(min(1.0, abs(float(a @ b)))))
               for i, a in enumerate(dirs) for b in dirs[i + 1:])
    if best < min_angle_deg:
        return None
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for t, d in rays:
        d = np.asarray(d, float)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ np.asarray(t, float)
    return np.linalg.solve(A, b)


# ───────── odom <-> world ────────────────────────────────────────────────────
class OdomWorld:
    """world = odom + offset. The EKF's odom frame starts at the spawn point."""

    def __init__(self, offset=(0.0, 0.0, 0.0)):
        self.offset = np.asarray(offset, float)

    @classmethod
    def from_pair(cls, world_xyz, odom_xyz) -> 'OdomWorld':
        """Offset measured from simultaneous ground-truth and EKF positions."""
        return cls(np.asarray(world_xyz, float) - np.asarray(odom_xyz, float))

    def odom_to_world(self, p) -> np.ndarray:
        return np.asarray(p, float) + self.offset

    def world_to_odom(self, p) -> np.ndarray:
        return np.asarray(p, float) - self.offset


def odom_to_world(p, offset) -> np.ndarray:
    return OdomWorld(offset).odom_to_world(p)


def world_to_odom(p, offset) -> np.ndarray:
    return OdomWorld(offset).world_to_odom(p)
