#!/bin/bash

set -euo pipefail

ACTION="${1:-status}"
BASE_DIR="${SEARXNG_BASE_DIR:-/path/to/searxng}"
IMAGE="${SEARXNG_IMAGE:-/path/to/searxng.sif}"
PORT="${SEARXNG_PORT:-18080}"
PID_FILE="$BASE_DIR/searxng.pid"
LOG_FILE="$BASE_DIR/logs/searxng.log"

is_running() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

start() {
    if is_running; then
        echo "SearXNG is already running (PID $(cat "$PID_FILE"))."
        return
    fi

    mkdir -p "$BASE_DIR/config" "$BASE_DIR/cache" "$BASE_DIR/logs"
    nohup env \
        SINGULARITYENV_SEARXNG_PORT="$PORT" \
        SINGULARITYENV_GRANIAN_HOST=0.0.0.0 \
        SINGULARITYENV_FORCE_OWNERSHIP=false \
        SINGULARITYENV_PYTHONPATH=/usr/local/searxng \
        singularity run \
        --bind "$BASE_DIR/config:/etc/searxng" \
        --bind "$BASE_DIR/cache:/var/cache/searxng" \
        "$IMAGE" >"$LOG_FILE" 2>&1 </dev/null &
    echo "$!" >"$PID_FILE"

    sleep 3
    if ! is_running; then
        echo "SearXNG failed to start. See $LOG_FILE" >&2
        exit 1
    fi
    echo "SearXNG started on port $PORT (PID $(cat "$PID_FILE"))."
}

stop() {
    if ! is_running; then
        rm -f "$PID_FILE"
        echo "SearXNG is not running."
        return
    fi

    kill "$(cat "$PID_FILE")"
    rm -f "$PID_FILE"
    echo "SearXNG stopped."
}

status() {
    if is_running; then
        echo "SearXNG is running (PID $(cat "$PID_FILE"), port $PORT)."
        curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null
        echo "Health check passed."
    else
        echo "SearXNG is not running."
        exit 1
    fi
}

case "$ACTION" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        start
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}" >&2
        exit 2
        ;;
esac
