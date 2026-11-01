#!/usr/bin/env bash
# =============================================================================
#  FloodScout: one command that runs everything and opens all windows.
#
#    scripts/floodscout_demo.sh            # seed 7, default request
#    scripts/floodscout_demo.sh 3          # another generated town
#    scripts/floodscout_demo.sh 7 "Find the people in the water first and mark a beacon for each."
#
#  Starts (if needed) the model server, generates/builds the town, launches Gazebo with its
#  GUI, the drone stack and the mission, records the overview cameras, and opens the live
#  FloodScout dashboard. Ctrl-C here (or q in the dashboard) stops everything.
#  Results: runs/live_seed<S>_<time>/  (report.md, map.png, metrics.json, recordings).
#  Afterwards, a video of the run:  python3 scripts/make_video.py --run <that folder>
# =============================================================================
set -o pipefail
cd "$(dirname "$0")/.."
WS="$PWD"
SEED="${1:-7}"
MISSION="${2:-Find everyone who is stranded, judge who needs help first, and mark a safe spot near each person for the rescue boat.}"
RUN="live_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
OUT="$WS/runs/$RUN"
export DISPLAY="${DISPLAY:-:0}"
source scripts/flood_env.sh
mkdir -p "$OUT/rec"

say() { printf '\033[1;35m[FloodScout]\033[0m %s\n' "$*"; }
PIDS=()
cleanup() {
  trap - INT TERM EXIT
  say "stopping ..."
  for p in "${PIDS[@]}"; do kill -INT "$p" 2>/dev/null; done
  sleep 2
  scripts/flood_stop.sh >/dev/null
  say "results in $OUT"
}
trap cleanup INT TERM EXIT

# 1) model server
if ! curl -s localhost:8080/health | grep -q ok; then
  [ -f "$HOME/models/qwen35/Qwen3.5-4B-Q4_K_M.gguf" ] || { say "model not found: run scripts/setup_vlm.sh first"; exit 1; }
  say "starting the model server (first start takes ~30 s) ..."
  setsid nohup scripts/run_vlm.sh > "$WS/runs/llama-server.log" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do curl -s localhost:8080/health | grep -q ok && break; sleep 2; done
  curl -s localhost:8080/health | grep -q ok || { say "model server did not start (runs/llama-server.log)"; exit 1; }
fi
say "model server ready"

# 2) world for this seed
scripts/flood_stop.sh >/dev/null
if [ ! -f "src/drone_basic/worlds/flood_town_${SEED}.sdf" ]; then
  say "generating town seed $SEED ..."
  python3 -m flood_vlm.scene_gen --seed "$SEED" --overview-cam
fi
colcon build --symlink-install --packages-select drone_basic flood_vlm > "$OUT/build.log" 2>&1 \
  || { say "build failed ($OUT/build.log)"; exit 1; }
source install/setup.bash

# 3) Gazebo (GUI) + drone + mission
say "launching Gazebo and the mission (seed $SEED) ..."
ros2 launch flood_vlm flood_mission.launch.py seed:="$SEED" headless:=false run_id:="$RUN" \
  runs_root:="$WS/runs" mission:="$MISSION" > "$OUT/launch.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 120); do gz service -l 2>/dev/null | grep -q "/world/flood_town/create" && break; sleep 1; done

# a good GUI viewpoint over the town (ignored if the GUI is not up yet)
( sleep 25; gz service -s /gui/move_to/pose --reqtype gz.msgs.GUICamera --reptype gz.msgs.Boolean --timeout 3000 \
    --req 'pose: {position: {x: -40, y: -56, z: 40}, orientation: {x: -0.115, y: 0.247, z: 0.413, w: 0.869}}' \
    >/dev/null 2>&1 ) &

# 4) overview recorder + live dashboard
python3 scripts/showcase_record.py --out "$OUT/rec" > "$OUT/rec.log" 2>&1 &
PIDS+=($!)
say "opening the dashboard (q to close, f for fullscreen) ..."
python3 scripts/floodscout_live.py --run "$OUT"
