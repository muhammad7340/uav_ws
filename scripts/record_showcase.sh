#!/usr/bin/env bash
# Run one headless mission and record it for the demo video / figures.
#   source scripts/flood_env.sh && scripts/record_showcase.sh 7
# Output: runs/showcase_seed<S>/ (mission report + VLM log) and runs/showcase_seed<S>/rec/ (frames).
# Then:   python3 scripts/make_video.py --run runs/showcase_seed<S>
set -o pipefail
cd "$(dirname "$0")/.."
SEED="${1:-7}"
RUN="showcase_seed${SEED}"
rm -rf "runs/$RUN"; mkdir -p "runs/$RUN/rec"
ros2 launch flood_vlm flood_mission.launch.py seed:="$SEED" headless:=true run_id:="$RUN" \
  runs_root:="$PWD/runs" exit_on_done:=true > "runs/$RUN/launch.log" 2>&1 &
LPID=$!
for _ in $(seq 1 90); do gz service -l 2>/dev/null | grep -q "/world/flood_town/create" && break; sleep 1; done
python3 scripts/showcase_record.py --out "runs/$RUN/rec" > "runs/$RUN/rec.log" 2>&1 &
RPID=$!
wait "$LPID"
kill -INT "$RPID" 2>/dev/null; sleep 2
scripts/flood_stop.sh
echo "recorded runs/$RUN"
