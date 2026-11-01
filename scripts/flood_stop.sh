#!/usr/bin/env bash
# Stop everything that belongs to the FloodScout stack and leave nothing orphaned.
#
# Matching by process name is not enough: when a `ros2 launch` dies, its nodes
# (drone_way_home, water_controller, bridges, ...) and the gz server can survive,
# and a second copy of drone_way_home makes the drone abort every goal
# ("cmd_vel override"). So this stops every process whose environment carries the
# flood partition set by scripts/flood_env.sh (GZ_PARTITION=flood_vlm), and nothing
# else - other simulations (e.g. Swishy) run without it. llama-server is kept.
PART="${FLOOD_GZ_PARTITION:-flood_vlm}"

# never stop our own callers (collect_floodsim.sh, mission_eval, ...): skip all ancestors
ancestors=" $$ "
p=$$
while [ "$p" -gt 1 ] 2>/dev/null; do
  p="$(awk '{print $4}' /proc/$p/stat 2>/dev/null)"; [ -z "$p" ] && break
  ancestors="$ancestors$p "
done

flood_pids() {
  for d in /proc/[0-9]*; do
    pid="${d#/proc/}"
    case "$ancestors" in *" $pid "*) continue ;; esac
    [ -r "$d/environ" ] || continue
    { tr '\0' '\n' < "$d/environ"; } 2>/dev/null | grep -qx "GZ_PARTITION=$PART" || continue
    cmd="$({ tr '\0' ' ' < "$d/cmdline"; } 2>/dev/null)"
    case "$cmd" in
      *llama-server*|*flood_stop*|"") continue ;;
      *eval.run_vqa*|*setup_vlm*) continue ;;        # VQA runs only talk to llama-server
      *shell-snapshots*) continue ;;   # agent/interactive shells that sourced flood_env.sh
    esac
    echo "$pid"
  done
}

pids="$(flood_pids)"
[ -z "$pids" ] && { echo "flood sim stopped (nothing running)"; exit 0; }
kill -INT $pids 2>/dev/null
for _ in $(seq 1 10); do
  pids="$(flood_pids)"; [ -z "$pids" ] && break; sleep 1
done
[ -n "$pids" ] && kill -TERM $pids 2>/dev/null && sleep 3
pids="$(flood_pids)"
[ -n "$pids" ] && kill -KILL $pids 2>/dev/null && sleep 1   # gz server ignores SIGTERM when orphaned
pids="$(flood_pids)"
if [ -n "$pids" ]; then
  echo "still running:"; ps -o pid,cmd -p $pids | cut -c1-120
  exit 1
fi
echo "flood sim stopped"
