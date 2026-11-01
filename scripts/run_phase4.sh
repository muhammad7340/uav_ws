#!/usr/bin/env bash
# =============================================================================
#  FloodScout Phase 4 queue (FLOOD_VLM_PLAN.md 4.2-4.4). Long: run detached.
#
#    source scripts/flood_env.sh
#    setsid nohup scripts/run_phase4.sh > results/phase4.log 2>&1 &
#
#  Every step is resumable (VQA logs per question, missions skip finished runs),
#  so the script can simply be started again after an interruption.
#  Order = most useful results first:
#    1. FloodSim-VQA test, Q4_K_M, 640x480          (baseline)
#    2. FloodNet VQA test (n=300), Q4_K_M, neutral prompt
#    3. FloodSim-VQA test, Q4_K_M, long side 448     (image-size ablation)
#    4. Missions seeds 1-3, look-again on / off
#    5. FloodSim-VQA test, Q8_0, 640                 (quantisation ablation; restarts llama-server)
#    6. Missions seeds 4-10, look-again on / off
#    7. ablation.csv + plots
# =============================================================================
set -o pipefail
cd "$(dirname "$0")/.."
N_SIM="${N_SIM:-120}"          # FloodSim test questions per config (20 per type)
VQA="python3 -u -m flood_vlm.eval.run_vqa"

restart_llama() {               # restart_llama Q4_K_M|Q8_0
  pkill -f "llama-serve[r]" || true
  sleep 3
  QUANT="$1" setsid nohup scripts/run_vlm.sh > "results/llama_$1.log" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do curl -s localhost:8080/health | grep -q ok && return 0; sleep 2; done
  echo "llama-server ($1) did not come up"; return 1
}

step() { echo; echo "=== $(date +%T) $* ==="; }

curl -s localhost:8080/health | grep -q ok || restart_llama Q4_K_M

step "1 FloodSim q4 640"
$VQA --dataset floodsim --n "$N_SIM" --quant q4 --tag q4_640
step "2 FloodNet q4 (224 px images)"
$VQA --dataset floodnet --n 300 --quant q4 --tag q4_224
step "3 FloodSim q4 448"
$VQA --dataset floodsim --n "$N_SIM" --quant q4 --long-side 448 --tag q4_448
step "4 missions seeds 1-3"
python3 -u -m flood_vlm.eval.mission_eval --seeds 1..3 --lookagain on off
step "5 FloodSim q8 640"
if [ -f "$HOME/models/qwen35/Qwen3.5-4B-Q8_0.gguf" ]; then
  restart_llama Q8_0 && $VQA --dataset floodsim --n "$N_SIM" --quant q8 --tag q8_640
  restart_llama Q4_K_M
fi
step "6 missions seeds 4-10"
python3 -u -m flood_vlm.eval.mission_eval --seeds 4..10 --lookagain on off
step "7 plots"
python3 -m flood_vlm.eval.plots
step "done"
