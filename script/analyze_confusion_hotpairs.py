"""3.1 Confusion matrix hot pair 분석.

visual-only baseline 의 confusion matrix 에서:
- per-class recall / precision
- symmetric confusion = (cm[i,j] + cm[j,i]) / (n_i + n_j) 로 hot pair top-K 선별
- pair plot heatmap 출력
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def shorten(s: str, n: int = 36) -> str:
    if " : " in s:
        s = s.split(" : ", 1)[0] + " : " + s.split(" : ", 1)[1]
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> None:
    run_dir = Path("/home/iw/dinov3_test/runs/exp2_cross_task_visual")
    summary = json.loads((run_dir / "analysis_exp2/summary.json").read_text())
    cm = np.asarray(summary["confusion_matrix"], dtype=np.int64)
    labels = summary["subtask_labels"]
    n = cm.shape[0]
    row_tot = cm.sum(axis=1)
    col_tot = cm.sum(axis=0)
    diag = np.diag(cm)
    recall = diag / np.maximum(row_tot, 1)
    precision = diag / np.maximum(col_tot, 1)

    print("=" * 80)
    print("Per-class recall / precision (visual-only baseline)")
    print("=" * 80)
    print(f"{'id':>3}  {'recall':>7}  {'prec':>7}  {'support':>7}  task")
    order = np.argsort(recall)
    for i in order:
        print(f"{i:>3}  {recall[i]:>7.3f}  {precision[i]:>7.3f}  {row_tot[i]:>7d}  {shorten(labels[str(i)], 60)}")

    # symmetric pairs
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            denom = row_tot[i] + row_tot[j]
            confused = cm[i, j] + cm[j, i]
            score = confused / max(denom, 1)
            pairs.append((score, confused, i, j))
    pairs.sort(reverse=True)

    print()
    print("=" * 80)
    print("Top-12 confused pairs (symmetric)")
    print("=" * 80)
    print(f"{'rate':>6}  {'count':>5}  pair")
    for score, cnt, i, j in pairs[:12]:
        print(f"{score:>6.3f}  {cnt:>5d}  ({i:>2}, {j:>2})  {shorten(labels[str(i)], 38)}  <->  {shorten(labels[str(j)], 38)}")

    # heatmap: row-normalized
    cm_row = cm / np.maximum(row_tot[:, None], 1)
    fig, ax = plt.subplots(1, 1, figsize=(11, 9))
    im = ax.imshow(cm_row, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    short_labels = [f"{i} {shorten(labels[str(i)], 28)}" for i in range(n)]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Row-normalized confusion (visual-only, knn={summary['knn_accuracy']:.3f})")
    for i in range(n):
        for j in range(n):
            v = cm_row[i, j]
            if v > 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    out = run_dir / "analysis_exp2/confusion_hotpairs.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n[saved] {out}")

    # save hot pair summary as JSON
    hot_pairs = [
        {"rate": float(s), "count": int(c), "i": int(i), "j": int(j),
         "label_i": labels[str(i)], "label_j": labels[str(j)]}
        for s, c, i, j in pairs[:12]
    ]
    out_json = run_dir / "analysis_exp2/hot_pairs.json"
    out_json.write_text(json.dumps({
        "per_class": [
            {"id": int(i), "recall": float(recall[i]), "precision": float(precision[i]),
             "support": int(row_tot[i]), "label": labels[str(i)]}
            for i in range(n)
        ],
        "hot_pairs": hot_pairs,
    }, indent=2))
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()
