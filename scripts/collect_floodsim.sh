#!/usr/bin/env bash
# =============================================================================
#  FloodSim-VQA frame collection for seeds 1..10 (FLOOD_VLM_PLAN.md 4.1).
#  For each seed: generate the world, start it headless (water schedule off),
#  fly the 3x3 grid at 20 m and 12 m at two water levels, stop the sim.
#
#  Usage:  source scripts/flood_env.sh
#          scripts/collect_floodsim.sh [first_seed] [last_seed]
#  then:   ros2 run flood_vlm floodsim_vqa_gen questions --check 20
# =============================================================================
set -o pipefail      # no -u: ROS setup.bash reads unset variables
cd "$(dirname "$0")/.."
FIRST="${1:-1}"
LAST="${2:-10}"
mkdir -p data/floodsim/logs
for seed in $(seq "$FIRST" "$LAST"); do
  if grep -q "\"seed\": $seed," data/floodsim/frames.jsonl 2>/dev/null; then
    echo "seed $seed already collected"; continue
  fi
  echo "=== seed $seed ==="
  python3 -m flood_vlm.scene_gen --seed "$seed"
  colcon build --symlink-install --packages-select drone_basic >/dev/null 2>&1   # install the new world
  source install/setup.bash
  ros2 launch flood_vlm flood_world.launch.py seed:="$seed" headless:=true water_schedule:=false \
    > "data/floodsim/logs/world_$seed.log" 2>&1 &
  sleep 15
  timeout 1800 python3 -m flood_vlm.eval.floodsim_vqa_gen collect --seed "$seed" \
    2>&1 | tee "data/floodsim/logs/collect_$seed.log" | grep -E "frames for seed|Error|error" || true
  scripts/flood_stop.sh >/dev/null
  sleep 3
done
