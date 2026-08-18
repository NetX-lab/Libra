#!/usr/bin/env bash
set -Eeuo pipefail

container="${EHP_DOCKER_CONTAINER:-ehp-container}"
[[ -n "${container}" ]] || { echo "EHP_DOCKER_CONTAINER is empty" >&2; exit 2; }
[[ "$#" -gt 0 ]] || { echo "usage: $0 command [args...]" >&2; exit 2; }

if [[ "${EHP_DOCKER_EXEC:-0}" != "1" ]]; then
    exec "$@"
fi

if [[ "${1:-}" == "bash" && "${2:-}" == "-lc" && "$#" -ge 3 ]]; then
    # The launcher passes `bash -lc <command>` for portability.  Do not spawn
    # a second login shell inside the image: its Ascend startup hook imports
    # torch before the requested command and can block process launch.
    command_line="$3"
else
    printf -v command_line '%q ' "$@"
fi
# The shared Ascend image's shell startup hooks import torch.  That can block
# before the requested command runs (and, during a migration, makes a healthy
# rollout look like a failed replacement).  Commands launched by this wrapper
# are fully specified, so bypass interactive/login startup files.
exec docker exec "$container" bash --noprofile --norc -c "${command_line}"
