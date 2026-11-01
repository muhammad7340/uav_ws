#!/usr/bin/env bash
# Stop a FloodScout-VLM simulation cleanly and make sure no gz server is orphaned.
# Only touches flood_town processes (never other Gazebo sims such as Swishy).
pkill -INT -f "ros2 launch flood_vl[m]" || true
for _ in $(seq 1 20); do pgrep -f "flood_tow[n]_" >/dev/null || break; sleep 1; done
pkill -TERM -f "ros2 launch flood_vl[m]" || true
pkill -TERM -f "gz sim .*flood_tow[n]_" || true
pkill -TERM -f "flood_vlm/lib/flood_vl[m]/mission_node" || true
for _ in $(seq 1 10); do pgrep -f "flood_tow[n]_" >/dev/null || break; sleep 1; done
pkill -KILL -f "gz sim .*flood_tow[n]_" || true     # gz server ignores SIGTERM when orphaned
sleep 1
pgrep -af "flood_tow[n]_" && echo "still running ^" || echo "flood sim stopped"
