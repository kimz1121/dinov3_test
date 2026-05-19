#!/usr/bin/env bash
# num_queries (learnable query 토큰 수) sweep.
#
# 사용:
#   bash script/sweep_num_queries.sh                       # K=4,8,16 기본
#   bash script/sweep_num_queries.sh --epochs 30           # contrastive_train.py 인자 전달
#   K_VALUES="2 4 8" bash script/sweep_num_queries.sh      # 다른 K 집합
#
# 결과 디렉토리:
#   runs/sweep_K_<timestamp>/K{4,8,16}/...

set -euo pipefail

TS=$(date +%Y%m%d_%H%M%S)
ROOT="runs/sweep_K_${TS}"
mkdir -p "$ROOT"

K_VALUES="${K_VALUES:-4 8 16}"

echo "[sweep] root=$ROOT  K_VALUES=$K_VALUES"
echo "[sweep] extra args: $*"
echo

for K in $K_VALUES; do
    OUT="$ROOT/K${K}"
    echo "=================================================="
    echo "[sweep] num_queries=$K -> $OUT"
    echo "=================================================="
    python script/contrastive_train.py \
        --num-queries "$K" \
        --out-dir "$OUT" \
        "$@"
done

echo
echo "[sweep] done. results under $ROOT"
