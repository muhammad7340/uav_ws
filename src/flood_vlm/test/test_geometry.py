"""Down-camera projection (FLOOD_VLM_PLAN.md 2.3)."""
import math

import numpy as np

from flood_vlm import geometry as G

W, H, HFOV, ALT = 640, 480, 1.3962634, 20.0


def _camera_at(x, y, z, yaw=0.0):
    q = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    return G.camera_pose_from_base((x, y, z), q)


def test_optical_axis_points_down():
    R, _ = _camera_at(0, 0, ALT)
    assert np.allclose(R @ [0, 0, 1], [0, 0, -1], atol=1e-3)


def test_centre_pixel_hits_point_below_drone():
    K = G.intrinsics_from_hfov(W, H, HFOV)
    R, t = _camera_at(3.0, -4.0, ALT)
    p = G.project_to_plane(W / 2, H / 2, K, R, t, 0.0)
    assert np.allclose(p[:2], [3.0, -4.0], atol=0.05)


def test_corner_offsets_match_hfov():
    K = G.intrinsics_from_hfov(W, H, HFOV)
    R, t = _camera_at(0, 0, ALT)
    h = ALT - 0.03                                  # optical centre is 3 cm below base_link
    half_w = h * math.tan(HFOV / 2)                 # ~16.8 m
    half_h = half_w * H / W                         # ~12.6 m
    p = G.project_to_plane(0, 0, K, R, t, 0.0)      # top-left pixel
    # Verified in Gazebo with a beacon: image up = world +x, image right = world -y.
    assert np.allclose(p[:2], [half_h, half_w], atol=0.1)
    p = G.project_to_plane(W, H, K, R, t, 0.0)      # bottom-right pixel
    assert np.allclose(p[:2], [-half_h, -half_w], atol=0.1)


def test_yaw_rotates_footprint():
    K = G.intrinsics_from_hfov(W, H, HFOV)
    R, t = _camera_at(0, 0, ALT, yaw=math.pi / 2)
    p = G.project_to_plane(W / 2, 0, K, R, t, 0.0)  # top-centre pixel: drone forward = +y now
    assert p[1] > 10 and abs(p[0]) < 0.1


def test_round_trip_and_rel_coords():
    K = G.intrinsics_from_hfov(W, H, HFOV)
    R, t = _camera_at(1.0, 2.0, ALT)
    u, v = G.rel_to_pixel(250, 750, W, H)
    p = G.project_to_plane(u, v, K, R, t, 3.0)
    assert abs(p[2] - 3.0) < 1e-9
    u2, v2 = G.world_to_pixel(p, K, R, t)
    assert abs(u2 - u) < 1e-6 and abs(v2 - v) < 1e-6
    assert G.pixel_to_rel(u2, v2, W, H) == (250.0, 750.0) or np.allclose(G.pixel_to_rel(u2, v2, W, H), (250, 750))


def test_odom_world_offset():
    ow = G.OdomWorld.from_pair((0.0, -26.0, 2.83), (0.0, 0.0, 0.0))
    assert np.allclose(ow.odom_to_world((10, 31, 17.17)), (10, 5, 20))
    assert np.allclose(ow.world_to_odom((10, 5, 20)), (10, 31, 17.17))
