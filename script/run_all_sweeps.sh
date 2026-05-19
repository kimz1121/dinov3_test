#!/usr/bin/env bash
# 4개 데이터셋 × K sweep 일괄 실행
set -euo pipefail

EPOCHS="${EPOCHS:-30}"
PROBE_EVERY="${PROBE_EVERY:-5}"
TS=$(date +%Y%m%d_%H%M%S)
ROOT="runs/all_sweeps_${TS}"
mkdir -p "$ROOT"

echo "[all_sweeps] root=$ROOT  epochs=$EPOCHS  probe_every=$PROBE_EVERY"

declare -A DATASETS=(
    [robocasa]="data/patch_embeddings"
    [robocasa_last]="data/patch_embeddings_last"
    [libero_goal_lerobot]="data/patch_embeddings_libero_goal_lerobot"
    [libero_goal_lerobot_last]="data/patch_embeddings_libero_goal_lerobot_last"
)

for DNAME in robocasa robocasa_last libero_goal_lerobot libero_goal_lerobot_last; do
    DROOT="${DATASETS[$DNAME]}"
    mkdir -p "$ROOT/$DNAME"
    echo ""
    echo "############################################################"
    echo "  DATASET: $DNAME  ($DROOT)"
    echo "############################################################"
    for K in 4 8 16; do
        OUT="$ROOT/${DNAME}/K${K}"
        LOG="$ROOT/${DNAME}/K${K}_run.log"
        echo ""
        echo "  --- K=${K} -> $OUT ---"
        python script/contrastive_train.py \
            --patch-h5-root "$DROOT" \
            --num-queries "$K" \
            --epochs "$EPOCHS" \
            --probe-every "$PROBE_EVERY" \
            --out-dir "$OUT" \
            2>&1 | tee "$LOG"
    done
done

echo ""
echo "[all_sweeps] ALL DONE. results under $ROOT"

# -------- 간단 요약 출력 --------
python - "$ROOT" <<'PYEOF'
import sys, json, pathlib

root = pathlib.Path(sys.argv[1])
rows = []
for log_file in sorted(root.glob("*/K*/log.jsonl")):
    dname = log_file.parent.parent.name
    K = int(log_file.parent.name.replace("K", ""))
    epoch_rows = [json.loads(l) for l in log_file.read_text().splitlines() if l]
    summaries = [r for r in epoch_rows if r.get("phase") == "epoch_summary"]
    probe_rows = [r for r in summaries if r.get("probe")]
    if not summaries:
        continue
    last = summaries[-1]
    last_probe = probe_rows[-1]["probe"] if probe_rows else {}
    rows.append({
        "dataset": dname, "K": K,
        "test_loss": last["test"]["loss"],
        "L_task": last["test"]["L_task"],
        "L_nuis": last["test"]["L_nuis"],
        **{k: round(v, 3) for k, v in last_probe.items()},
    })

rows.sort(key=lambda r: (r["dataset"], r["K"]))
header = f"{'dataset':<30} {'K':>3} {'test_loss':>10} {'L_task':>8} {'L_nuis':>8} {'task@zt':>8} {'cam@zt':>7} {'cam@zn':>7} {'task@zn':>8}"
print("\n" + "=" * len(header))
print(header)
print("=" * len(header))
prev_d = None
for r in rows:
    if r["dataset"] != prev_d:
        if prev_d is not None:
            print()
        prev_d = r["dataset"]
    print(
        f"{r['dataset']:<30} {r['K']:>3}"
        f"  {r['test_loss']:>9.4f}  {r['L_task']:>7.4f}  {r['L_nuis']:>7.4f}"
        f"  {r.get('task_on_z_task', float('nan')):>7.3f}"
        f"  {r.get('cam_on_z_task', float('nan')):>6.3f}"
        f"  {r.get('cam_on_z_nuis', float('nan')):>6.3f}"
        f"  {r.get('task_on_z_nuis', float('nan')):>7.3f}"
    )
print("=" * len(header))
PYEOF
