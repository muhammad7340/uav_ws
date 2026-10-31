"""Load config/mission.yaml from the installed share dir or the source tree."""
from __future__ import annotations

import os

import yaml

_HERE = os.path.dirname(os.path.realpath(__file__))   # src tree, also under --symlink-install


def share_dir() -> str:
    """Installed share/flood_vlm, or the package source dir when not installed."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory('flood_vlm')
    except Exception:
        return os.path.dirname(_HERE)


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(share_dir(), 'config', 'mission.yaml')
        if not os.path.exists(path):
            path = os.path.join(os.path.dirname(_HERE), 'config', 'mission.yaml')
    with open(path) as f:
        return yaml.safe_load(f)


def drone_worlds_dir() -> str:
    """Where generated worlds go: drone_basic/worlds in the source tree.

    Falls back to the installed share dir if the source tree is not found.
    """
    src = os.path.normpath(os.path.join(_HERE, '..', '..', 'drone_basic', 'worlds'))
    if os.path.isdir(src):
        return src
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('drone_basic'), 'worlds')
