"""Exp 2 cross-task (13-class) 분석 — confusion / episode_gram / sample_gram / tsne_grid.

contrastive_train_cross_task.py 또는 contrastive_train_cross_task_vl.py
가 만든 run-dir 들에 대해 동일한 4 종 시각화 생성.

사용:
    python script/analyze_exp2.py \\
        --run-dirs runs/exp2_cross_task_visual \\
                   runs/exp2_cross_task_vl_alignment_infonce \\
                   runs/exp2_cross_task_vl_anchor_regression
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_exp1 import (
    collect_z,
    knn_predict,
    confusion_matrix,
    per_episode_mean,
    plot_tsne_grid,
    shorten,
)
from contrastive_train import DisentangleModel, PatchClipDataset
from contrastive_train_cross_task import build_cross_task_indices


# ---------------------------------------------------------------------------
# 13-class confusion / Gram 시각화
# ---------------------------------------------------------------------------


def plot_confusion_cross(
    cm: np.ndarray,
    labels: list[str],
    knn_k: int,
    accuracy: float,
    out_path: Path,
) -> None:
    """13×13 confusion matrix (row-normalized).

    레이아웃: axes 에 id 만 ("0", "1", ..., "12"). caption 에 instruction 전체.
    """
    n = cm.shape[0]
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")

    # cell 값: count + (normalized)
    for i in range(n):
        for j in range(n):
            color = "white" if norm[i, j] > 0.5 else "black"
            ax.text(
                j, i,
                f"{cm[i, j]}\n{norm[i, j]:.2f}",
                ha="center", va="center",
                fontsize=7, color=color,
            )

    ax.set_xticks(range(n))
    ax.set_xticklabels([str(i) for i in range(n)], fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(i) for i in range(n)], fontsize=10)
    ax.set_xlabel("Predicted sub-task id")
    ax.set_ylabel("True sub-task id")
    ax.set_title(
        f"Cross-task 13-class k-NN confusion (test, k={knn_k}, acc={accuracy:.3f})",
        fontsize=12,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("row-normalized")

    # caption: 각 sub-task 의 instruction
    legend_text = "\n".join(f"{i:>2}  =  {labels[i]}" for i in range(n))
    fig.text(
        0.5, 0.01, legend_text,
        ha="center", va="bottom",
        fontsize=8, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="none"),
    )

    plt.subplots_adjust(bottom=0.32)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_episode_gram_cross(
    gram: np.ndarray,
    ep_subtasks_sorted: np.ndarray,
    subtask_labels: dict[int, str],
    stats: dict[str, float],
    out_path: Path,
) -> None:
    """13-block episode×episode Gram (color strip + boundaries)."""
    from matplotlib.patches import Patch

    n = gram.shape[0]
    unique_st = sorted(set(int(x) for x in ep_subtasks_sorted))
    cmap_cat = plt.get_cmap("tab20")
    color_map = {st: cmap_cat(i % 20) for i, st in enumerate(unique_st)}

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[0.025, 1.0, 0.04],
        height_ratios=[0.025, 1.0],
        wspace=0.02, hspace=0.02,
        left=0.05, right=0.93, top=0.92, bottom=0.40,
    )
    ax_top = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])

    im = ax_main.imshow(gram, cmap="bwr", vmin=-1, vmax=1, aspect="auto")
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    for b in np.where(np.diff(ep_subtasks_sorted) != 0)[0]:
        ax_main.axvline(b + 0.5, color="k", linewidth=0.8, alpha=0.6)
        ax_main.axhline(b + 0.5, color="k", linewidth=0.8, alpha=0.6)

    top_strip = np.array(
        [list(color_map[int(s)]) for s in ep_subtasks_sorted], dtype=float
    ).reshape(1, n, 4)
    ax_top.imshow(top_strip, aspect="auto")
    ax_top.set_xticks([]); ax_top.set_yticks([])
    ax_left.imshow(np.swapaxes(top_strip, 0, 1), aspect="auto")
    ax_left.set_xticks([]); ax_left.set_yticks([])

    fig.colorbar(im, cax=ax_cbar, label="cosine similarity")
    fig.suptitle(
        "Cross-task: episode × episode Gram of z_task (test, sorted by sub-task)",
        fontsize=12, y=0.97,
    )

    legend_patches = [
        Patch(color=color_map[st], label=f"{st:>2}  {shorten(subtask_labels[st], 50)}")
        for st in unique_st
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=2, frameon=False, fontsize=8,
    )

    stat_text = (
        f"within sub-task cos = {stats['within']:.4f}   "
        f"cross sub-task cos = {stats['cross']:.4f}   "
        f"delta = {stats['within'] - stats['cross']:+.4f}   "
        f"(test episodes = {n})"
    )
    fig.text(0.5, 0.34, stat_text, ha="center", va="bottom",
             fontsize=10, family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="none"))

    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sample_gram_cross(
    Z_task: np.ndarray,
    Z_nuis: np.ndarray,
    sid: np.ndarray,
    cid: np.ndarray,
    ep_idx: np.ndarray,
    subtask_labels: dict[int, str],
    out_path: Path,
) -> None:
    """Sample×sample cosine, 13 sub-task block, compact."""
    sort_key = np.lexsort((cid, ep_idx, sid))
    Zt = Z_task[sort_key]
    Zn = Z_nuis[sort_key]
    sid_s = sid[sort_key]

    gram_t = Zt @ Zt.T
    gram_n = Zn @ Zn.T
    n = gram_t.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8.5))
    boundaries_sub = np.where(np.diff(sid_s) != 0)[0]

    for ax, gram, label in [
        (axes[0], gram_t, "z_task"),
        (axes[1], gram_n, "z_nuis"),
    ]:
        off_diag = gram[~np.eye(n, dtype=bool)]
        vlo, vhi = np.percentile(off_diag, [2, 98])
        vrange = max(abs(vlo), abs(vhi), 0.1)
        im = ax.imshow(gram, cmap="bwr", vmin=-vrange, vmax=vrange,
                        interpolation="nearest", aspect="equal")
        for b in boundaries_sub:
            ax.axvline(b + 0.5, color="k", linewidth=0.8, alpha=0.7)
            ax.axhline(b + 0.5, color="k", linewidth=0.8, alpha=0.7)
        ax.set_title(label, fontsize=14, fontweight="bold", pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")

        # sub-task id 라벨 (블록 중앙)
        for st in sorted(set(int(x) for x in sid_s)):
            m = sid_s == st
            pos = float(np.where(m)[0].mean())
            ax.text(pos, -n * 0.015, f"{st}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle(
        f"Cross-task: sample × sample cosine (n={n})  ·  sorted by sub-task → episode → camera",
        fontsize=12, y=0.97,
    )

    legend_text = "\n".join(
        f"{k:>2}  =  {shorten(subtask_labels[k], 70)}"
        for k in sorted(subtask_labels.keys())
    )
    fig.text(0.5, 0.02, legend_text, ha="center", va="bottom",
             fontsize=8, family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="none"))

    plt.subplots_adjust(top=0.92, bottom=0.30, left=0.05, right=0.97, wspace=0.20)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-run
# ---------------------------------------------------------------------------


def analyze_run(run_dir: Path, ckpt_name: str, knn_k: int, device: str) -> dict:
    out = run_dir / "analysis_exp2"
    out.mkdir(exist_ok=True)

    config = json.loads((run_dir / "config.json").read_text())
    subtask_to_instr = {int(k): v for k, v in config["subtask_id_to_label"].items()}
    n_cls = int(config["num_subtasks"])

    mc = config["model_config"]
    model = DisentangleModel(**mc).to(device)
    ckpt = torch.load(run_dir / ckpt_name, map_location=device, weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    elif "vl_model" in ckpt:
        vl_sd = ckpt["vl_model"]
        visual_sd = {k[len("visual."):]: v for k, v in vl_sd.items() if k.startswith("visual.")}
        model.load_state_dict(visual_sd)
    else:
        raise SystemExit(f"unknown ckpt keys: {list(ckpt.keys())}")

    h5_paths = [Path(p) for p in config["h5_paths"]]
    seed = int(config["args"]["seed"])
    rng = np.random.default_rng(seed)
    train_idx, test_idx, cams, _ = build_cross_task_indices(
        h5_paths,
        Path(config["args"]["patch_h5_root"]),
        config["unified_mapping"],
        float(config["args"]["train_ratio"]),
        config["args"].get("cameras"),
        rng,
    )
    train_ds = PatchClipDataset(train_idx)
    test_ds = PatchClipDataset(test_idx)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    Zt_tr, _, sid_tr, _, _ = collect_z(model, train_loader, device)
    Zt_te, Zn_te, sid_te, cid_te, ep_te = collect_z(model, test_loader, device)

    # A. Confusion (13×13)
    pred_te = knn_predict(Zt_tr, sid_tr, Zt_te, k=knn_k)
    cm = confusion_matrix(sid_te, pred_te, n_cls)
    acc = float((pred_te == sid_te).mean())
    plot_confusion_cross(
        cm, [subtask_to_instr[i] for i in range(n_cls)],
        knn_k, acc, out / "confusion_matrix.png",
    )

    # D. Episode Gram (test set, per-ep mean)
    # ep_idx 가 task 별로 0..N 중복이므로 (task, ep) 조합으로 묶어야 함.
    # SampleIndex.h5_path 가 다르므로 (h5_path, ep_idx) 로 unique 화. 여기선 단순화:
    # test_idx 의 h5_path 별로 ep_idx 가 unique 라 가정하고, (h5_path + ep) key 사용.
    h5_per_sample = np.array(
        [test_idx[i].h5_path for i in range(len(test_idx))]
    )
    ep_keys = np.array([f"{h}_{e}" for h, e in zip(h5_per_sample, ep_te)])
    unique_keys, inverse = np.unique(ep_keys, return_inverse=True)
    ep_mean_list = []
    ep_subtasks = []
    for k_id in range(len(unique_keys)):
        m = inverse == k_id
        v = Zt_te[m].mean(0)
        v = v / max(np.linalg.norm(v), 1e-8)
        ep_mean_list.append(v)
        ep_subtasks.append(int(sid_te[m][0]))
    ep_mean = np.array(ep_mean_list)
    ep_subtasks = np.array(ep_subtasks)
    order = np.argsort(ep_subtasks, kind="stable")
    ep_mean_s = ep_mean[order]
    ep_subtasks_s = ep_subtasks[order]
    gram_ep = ep_mean_s @ ep_mean_s.T

    within, cross = [], []
    nep = gram_ep.shape[0]
    for i in range(nep):
        for j in range(i + 1, nep):
            (within if ep_subtasks_s[i] == ep_subtasks_s[j] else cross).append(float(gram_ep[i, j]))
    w_mean = float(np.mean(within)) if within else 0.0
    c_mean = float(np.mean(cross)) if cross else 0.0
    plot_episode_gram_cross(
        gram_ep, ep_subtasks_s, subtask_to_instr,
        {"within": w_mean, "cross": c_mean},
        out / "episode_gram.png",
    )

    # B. Sample Gram (sample×sample)
    plot_sample_gram_cross(
        Zt_te, Zn_te, sid_te, cid_te, ep_te,
        subtask_to_instr, out / "sample_gram.png",
    )

    # C. t-SNE 4 panel
    plot_tsne_grid(
        Zt_te, Zn_te, sid_te, cid_te,
        subtask_to_instr, cams,
        config.get("experiment", "exp2"),
        out / "tsne_grid.png",
    )

    result = {
        "experiment": config.get("experiment"),
        "num_subtasks": n_cls,
        "knn_accuracy": acc,
        "within_subtask_cos": w_mean,
        "cross_subtask_cos": c_mean,
        "within_minus_cross": w_mean - c_mean,
        "confusion_matrix": cm.tolist(),
        "subtask_labels": {str(k): v for k, v in subtask_to_instr.items()},
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dirs", nargs="+", required=True, type=Path)
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--summary-out", type=Path,
                   default=Path("runs/exp2_summary.json"))
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}  knn_k={args.knn_k}")

    summary_all = []
    for rd in args.run_dirs:
        if not rd.exists():
            print(f"[skip] {rd}")
            continue
        print(f"\n=== {rd} ===")
        r = analyze_run(rd, args.ckpt, args.knn_k, device)
        r["run"] = rd.name
        summary_all.append(r)
        print(f"  knn_acc={r['knn_accuracy']:.3f}  "
              f"within={r['within_subtask_cos']:.3f}  cross={r['cross_subtask_cos']:.3f}  "
              f"Δ={r['within_minus_cross']:+.3f}")

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary_all, indent=2, ensure_ascii=False))
    print(f"\n[done] summary → {args.summary_out}")


if __name__ == "__main__":
    main()
