#!/bin/bash
# =============================================================================
# Search-R1 + Qwen3-14B + SearXNG production training
#
# Topology:
#   node 0: 4-GPU FSDP training
#   node 1: 4-GPU TP=4 rollout
#   node 2: two 2-GPU TP=2 rollout instances
#
# This wrapper reuses the validated C-MLFQ launcher. Before allocating model
# memory it verifies the official Search-R1 data and requires SearXNG to return
# at least one real web result.
#
# Submit:
#   MODEL_PATH=/path/to/Qwen3-14B \
#   sbatch --export=ALL,MODEL_PATH scripts/submit_search_r1_searxng_train.slurm
#
# Common overrides:
#   TOTAL_STEPS=5000 TRAIN_BATCH_SIZE=8 MAX_NEW_TOKENS=1024 \
#   SEARXNG_URL=http://10.32.44.56:18080 \
#   sbatch --export=ALL,MODEL_PATH,TOTAL_STEPS,TRAIN_BATCH_SIZE,MAX_NEW_TOKENS,SEARXNG_URL \
#     scripts/submit_search_r1_searxng_train.slurm
# =============================================================================

#SBATCH -J search_r1_searxng
#SBATCH -o logs/search_r1_searxng_%j.out
#SBATCH -e logs/search_r1_searxng_%j.err
#SBATCH -N 3
#SBATCH -G 12
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH -p gpu
#SBATCH --time=7-00:00:00
#SBATCH --exclude=gn019

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/RL_Framework}"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-14B}"
SEARXNG_URL="${SEARXNG_URL:-http://10.32.44.56:18080}"
SEARCH_R1_DATA="${SEARCH_R1_DATA:-$PROJECT_DIR/data/search_r1_train.jsonl}"
SEARCH_PREFLIGHT_QUERY="${SEARCH_PREFLIGHT_QUERY:-PyTorch distributed training}"

export SEARCH_BACKEND=searxng
export SEARXNG_URL
export MODEL_PATH
export TOTAL_STEPS="${TOTAL_STEPS:-1000}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
export TRAIN_MAX_SEQ_LENGTH="${TRAIN_MAX_SEQ_LENGTH:-40960}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export MAX_WAIT="${MAX_WAIT:-1200}"

cd "$PROJECT_DIR"
mkdir -p logs

echo "============================================================"
echo "Search-R1 + SearXNG production training preflight"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID}"
echo "Model: $MODEL_PATH"
echo "Dataset: $SEARCH_R1_DATA"
echo "SearXNG: $SEARXNG_URL"
echo "Steps: $TOTAL_STEPS"
echo "Batch size: $TRAIN_BATCH_SIZE"
echo "Max sequence length: $TRAIN_MAX_SEQ_LENGTH"
echo "Max new tokens per turn: $MAX_NEW_TOKENS"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: model path does not exist: $MODEL_PATH" >&2
    exit 2
fi

python3 - "$SEARCH_R1_DATA" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"ERROR: Search-R1 dataset does not exist: {path}")

counts = Counter()
rows = 0
with path.open(encoding="utf-8") as source:
    for line in source:
        row = json.loads(line)
        counts[row["data_source"]] += 1
        rows += 1

expected = {"nq", "hotpotqa"}
if set(counts) != expected:
    raise SystemExit(
        f"ERROR: expected Search-R1 sources {sorted(expected)}, got {dict(counts)}"
    )
if rows != 169615:
    raise SystemExit(f"ERROR: expected 169615 Search-R1 rows, got {rows}")
print(f"Search-R1 data OK: {rows} rows, sources={dict(sorted(counts.items()))}")
PY

curl -fsS --max-time 10 "$SEARXNG_URL/healthz" >/dev/null
python3 - "$SEARXNG_URL" "$SEARCH_PREFLIGHT_QUERY" <<'PY'
import json
import sys
import urllib.parse
import urllib.request

base_url, query = sys.argv[1:]
url = base_url.rstrip("/") + "/search?" + urllib.parse.urlencode(
    {"q": query, "format": "json", "language": "en-US"}
)
with urllib.request.urlopen(url, timeout=45) as response:
    payload = json.load(response)

results = payload.get("results", [])
if not results:
    engines = payload.get("unresponsive_engines", [])
    raise SystemExit(
        "ERROR: SearXNG returned zero results. Real Search-R1 training is "
        f"blocked; check the server's outbound proxy. Unresponsive={engines}"
    )
print(f"SearXNG real-search OK: query={query!r}, results={len(results)}")
print(f"First result: {results[0].get('title', '')}")
PY

echo "Preflight passed. Starting the validated three-node training launcher."
exec bash "$PROJECT_DIR/scripts/submit_qwen3_cmlfq_train.slurm"
