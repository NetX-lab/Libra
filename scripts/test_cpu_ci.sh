#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q --strict-markers --junitxml=reports/cpu-tests.xml tests/
