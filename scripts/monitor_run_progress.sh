#!/usr/bin/env bash
# Read-only progress monitor for a detached formal run.
set -u

launcher_pid="${1:?usage: $0 LAUNCHER_PID RUN_DIR HISTORY_GLOB [INTERVAL]}"
run_dir="${2:?usage: $0 LAUNCHER_PID RUN_DIR HISTORY_GLOB [INTERVAL]}"
history_glob="${3:?usage: $0 LAUNCHER_PID RUN_DIR HISTORY_GLOB [INTERVAL]}"
interval="${4:-50}"

while true; do
    alive=0
    kill -0 "$launcher_pid" 2>/dev/null && alive=1
    history_files=()
    while IFS= read -r path; do history_files+=("$path"); done < <(
        compgen -G "$history_glob" || true
    )
    steps=0
    if ((${#history_files[@]})); then
        steps="$(wc -l "${history_files[@]}" | awk 'END {print $1 + 0}')"
    fi
    error_count="$(grep -h -E -c 'Traceback|RuntimeError|ERROR' \
        "$run_dir"/driver_node_*.log 2>/dev/null | awk '{sum += $1} END {print sum + 0}')"
    last_grp="$(grep -h -E 'GlobalResourcePlanner.*step=' \
        "$run_dir"/driver_node_0.log 2>/dev/null | tail -n 1 || true)"
    printf '%s alive=%s history_steps=%s errors=%s grp=%s\n' \
        "$(date '+%F %T')" "$alive" "$steps" "$error_count" "$last_grp"
    if [[ "$alive" -eq 0 || "$steps" -ge 100 ]]; then
        break
    fi
    sleep "$interval"
done
