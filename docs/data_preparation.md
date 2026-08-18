# Data Preparation

The commands below assume the Libra repository root is the current directory.

## R2E-Gym

Place the official R2E-Gym V1 parquet shards in one directory, then build the
validated task index:

```bash
python data/prepare_r2e_gym.py data/r2e_gym_v1
```

The full workflow also needs the Docker images referenced by the dataset.

## Search-R1

Download the official Search-R1 training parquet and convert it to Libra's local
schema:

```bash
python data/prepare_search_r1.py \
  /path/to/train.parquet \
  data/search_r1_train.jsonl
```

Configure one search backend:

```bash
export SEARXNG_URL=http://searxng-host:8080
# or
export SERPER_KEY_ID=/path/to/serper_key_file
```

## DAPO-Math-17K

Place the official parquet or JSONL file under `data/DAPO-Math-17K/{all,en}/`,
or point Libra to it explicitly:

```bash
export DAPO_MATH_PATH=/path/to/train.parquet
```

Continue with the [cluster manual](manual.md) to configure and launch a
training workload.
