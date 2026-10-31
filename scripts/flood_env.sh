#!/usr/bin/env bash
# Source this before any FloodScout-VLM command:   source scripts/flood_env.sh
# Isolates ROS 2 and Gazebo traffic from other simulations on the same machine
# (e.g. the Swishy aquarium) and adds the flood models to the Gazebo path.
_ws="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
[ -f "$_ws/install/setup.bash" ] && source "$_ws/install/setup.bash"
export ROS_DOMAIN_ID="${FLOOD_ROS_DOMAIN_ID:-42}"
export GZ_PARTITION="${FLOOD_GZ_PARTITION:-flood_vlm}"
export GZ_SIM_RESOURCE_PATH="$_ws/src/flood_vlm/models:$HOME/gazebo_models:${GZ_SIM_RESOURCE_PATH:-}"
unset _ws
