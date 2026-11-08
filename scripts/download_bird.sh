#!/usr/bin/env bash
# Download BIRD into data/bird/{train,dev}. train.zip is ~8.9 GB; dev.zip ~350 MB.
set -euo pipefail
ROOT=${1:-data/bird}
mkdir -p "$ROOT" && cd "$ROOT"
BASE=https://bird-bench.oss-cn-beijing.aliyuncs.com

if [ ! -d dev ]; then
  curl -L -o dev.zip "$BASE/dev.zip" && unzip -q dev.zip && rm dev.zip
  mv dev_20240627 dev && (cd dev && unzip -q dev_databases.zip && rm dev_databases.zip)
fi
if [ "${SKIP_TRAIN:-0}" != "1" ] && [ ! -d train ]; then
  curl -L -o train.zip "$BASE/train.zip" && unzip -q train.zip && rm train.zip
  (cd train && unzip -q train_databases.zip && rm train_databases.zip)
fi
echo "BIRD ready under $ROOT"
