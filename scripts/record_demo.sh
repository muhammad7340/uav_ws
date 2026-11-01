#!/usr/bin/env bash
# =============================================================================
#  FloodScout demo recording (FLOOD_VLM_PLAN.md 3.4)
#
#  Usage:  source scripts/flood_env.sh
#          scripts/run_vlm.sh &                         # llama-server
#          scripts/record_demo.sh [seed] [out_dir]
#
#  Starts the mission WITH the Gazebo GUI, points the GUI camera at the town,
#  records a rosbag (down camera, /flood/*, ground truth) and an ffmpeg screen
#  capture. The sim pauses while the VLM thinks: cut those parts in editing or
#  overlay the reasoning text from runs/<id>/vlm_calls.jsonl ("VLM thinking...").
#  Ctrl-C stops everything.
# =============================================================================
set -euo pipefail
SEED="${1:-7}"
OUT="${2:-runs/demo_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
export DISPLAY="${DISPLAY:-:0}"

pids=()
cleanup() {
  echo "stopping..."
  for p in "${pids[@]}"; do kill -INT "$p" 2>/dev/null || true; done
  sleep 3
  "$(dirname "$0")/flood_stop.sh" || true
}
trap cleanup EXIT INT TERM

ros2 launch flood_vlm flood_mission.launch.py seed:="$SEED" headless:=false \
  runs_root:="$PWD/$OUT" run_id:=mission > "$OUT/launch.log" 2>&1 &
pids+=($!)

# wait for the GUI, then move its camera over the town (follow the drone with
# the GUI's "Follow" on drone_custom if you prefer a chase view)
for _ in $(seq 1 120); do gz service -l 2>/dev/null | grep -q "/gui/move_to/pose" && break; sleep 1; done
sleep 20
gz service -s /gui/move_to/pose --reqtype gz.msgs.GUICamera --reptype gz.msgs.Boolean --timeout 3000 \
  --req 'pose: {position: {x: -38, y: -48, z: 42}, orientation: {x: -0.13, y: 0.30, z: 0.38, w: 0.86}}' || true

ros2 bag record -o "$OUT/bag" /camera_down/image /camera_down/camera_info /drone/odom_ground_truth \
  /flood/water_level /flood/rescue_requests /flood/markers /flood/events /drone/state > "$OUT/bag.log" 2>&1 &
pids+=($!)

SIZE="$(xdpyinfo 2>/dev/null | awk '/dimensions/{print $2}' || echo 1920x1080)"
ffmpeg -loglevel error -y -f x11grab -framerate 15 -video_size "$SIZE" -i "$DISPLAY" \
  -c:v libx264 -preset veryfast -crf 26 -pix_fmt yuv420p "$OUT/screen.mp4" &
pids+=($!)

echo "recording to $OUT (Ctrl-C to stop); mission log: $OUT/launch.log"
# stop when the mission node exits
while pgrep -f "flood_vlm/lib/flood_vl[m]/mission_node" >/dev/null || [ ! -f "$OUT/mission/metrics.json" ]; do
  sleep 5
  grep -q "mission finished" "$OUT/launch.log" 2>/dev/null && break
done
sleep 5
