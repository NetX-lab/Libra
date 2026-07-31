# Data Preparation

The commands below assume that Libra is installed with `python -m pip install
-e .` and that the repository root is the current directory. Dataset payloads
are excluded from Git; the download and conversion scripts remain
source-controlled.

The upstream datasets are not redistributed under Libra's MIT license. Review
and comply with each dataset's license and terms before downloading or
publishing derived artifacts.

## Common prerequisites

The base installation provides `datasets`, `huggingface_hub`, and `pyarrow`.
Authenticate only when an upstream repository requires it:

```bash
source .venv/bin/activate
hf auth login
```

Generated parquet, JSONL, index, and manifest files are ignored by Git.

## R2E-Gym V1

Source: [R2E-Gym/R2E-Gym-V1](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-V1)

Download the 13 official parquet shards and place them directly under
`data/r2e_gym_v1`:

```bash
mkdir -p data/r2e_gym_v1 /tmp/r2e_gym_v1
hf download R2E-Gym/R2E-Gym-V1 \
  --repo-type dataset \
  --include "data/train-*.parquet" \
  --local-dir /tmp/r2e_gym_v1
cp /tmp/r2e_gym_v1/data/train-*.parquet data/r2e_gym_v1/
```

Build the compact index and checksum manifest:

```bash
python data/prepare_r2e_gym.py data/r2e_gym_v1
python examples/test_r2e_gym.py
```

The validator expects:

- 13 files named `train-00000-of-00013.parquet` through
  `train-00012-of-00013.parquet`;
- 8,101 tasks from the 13 supported repositories;
- the columns used by the R2E workflow, including `docker_image`,
  `commit_hash`, `prompt`, `problem_statement`, and `expected_output_json`.

Successful preparation writes:

```text
data/r2e_gym_v1/index.jsonl
data/r2e_gym_v1/manifest.json
```

`manifest.json` records the row counts, repository counts, file sizes, and
SHA-256 checksum of every shard. Set `R2E_GYM_INDEX` only when storing the
generated index elsewhere:

```bash
export R2E_GYM_INDEX=/shared/data/r2e_gym_v1/index.jsonl
```

R2E execution also requires access to every `docker_image` referenced in the
dataset. Pull those images before allocating a long job and verify that the
compute nodes can run the configured Docker-, Podman-, or site-specific
container runtime. The dataset validator checks image names but does not pull
images.

## Search-R1

Upstream project: [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1)

Training data used by Libra:
[Archive-models/nq_hotpotqa_train_search](https://huggingface.co/datasets/Archive-models/nq_hotpotqa_train_search)

Materialize the upstream dataset as one parquet file:

```bash
mkdir -p data/search_r1_raw
python - <<'PY'
from datasets import load_dataset

dataset = load_dataset(
    "Archive-models/nq_hotpotqa_train_search",
    split="train",
)
dataset.to_parquet("data/search_r1_raw/train.parquet")
print(dataset)
PY
```

Convert it to Libra's local schema:

```bash
python data/prepare_search_r1.py \
  data/search_r1_raw/train.parquet \
  data/search_r1_train.jsonl
```

The converter requires `id`, `question`, `golden_answers`, and `data_source`,
accepts the `nq` and `hotpotqa` sources, and validates the expected 169,615
rows. For a deliberately filtered derivative, pass `--expected-rows 0` and
record the filtering procedure separately.

Configure exactly one search backend.

For SearXNG:

```bash
export SEARCH_BACKEND=searxng
export SEARXNG_URL=http://searxng-host:18080
curl -fsS "$SEARXNG_URL/healthz"
curl -fsS --get "$SEARXNG_URL/search" \
  --data-urlencode "q=OpenAI" \
  --data-urlencode "format=json"
```

For Serper:

```bash
export SEARCH_BACKEND=serper
export SERPER_KEY_ID=your_serper_api_key
```

`SERPER_KEY_ID` contains the API key itself, not a path to a key file. Never
commit it. The production SearXNG launcher performs a non-empty-result
preflight before training.

## DAPO-Math-17K

Source:
[BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)

Download the official parquet into Libra's default discovery layout:

```bash
mkdir -p data/DAPO-Math-17K/all
python - <<'PY'
from datasets import load_dataset

dataset = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
dataset.to_parquet(
    "data/DAPO-Math-17K/all/train-00000-of-00001.parquet"
)
print(dataset)
PY
```

Libra searches for `train.jsonl` or `train-00000-of-00001.parquet` under the
selected split. The relevant controls are:

```bash
export DAPO_MATH_ROOT="$PWD/data/DAPO-Math-17K"
export DAPO_MATH_SPLIT=all       # all or en
# Alternatively bypass discovery:
# export DAPO_MATH_PATH=/shared/data/dapo-math-17k.parquet
```

Validate usable prompts, ground truths, and prompt IDs before submitting:

```bash
DAPO_MIN_ROWS=17000 python examples/test_dapo_math_data.py
```

Runtime preprocessing accepts the upstream `prompt`, `reward_model`,
`extra_info`, and `solution` fields, normalizes them to Libra's schema, and
deduplicates by `prompt_id` unless `DAPO_DEDUPLICATE=0`.

Continue with the [cluster manual](manual.md) to configure and launch a
training workload.
