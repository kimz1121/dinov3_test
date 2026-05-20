"""Exp 1a sub-task 분석 — A (confusion matrix) + C (summary table) + D (episode Gram).

입력: contrastive_train_subtask.py 가 만든 run-dir (들)
출력 (각 run-dir 안):
    analysis_exp1/
      confusion_matrix.png   (A) k-NN 분류 confusion matrix (2×2)
      episode_gram.png       (D) episode × episode cosine Gram (sub-task 정렬)
      summary.json           per-task 핵심 metric
종합:
    --summary-out 지정 경로(default: runs/exp1_summary.json) 에 모든 task summary 합본
    + 같은 stem 의 .png 에 막대그래프 (k-NN acc vs majority class)

사용 예:
    python script/analyze_exp1.py \\
        --run-dirs runs/exp1_CloseDrawer_visual \\
                   runs/exp1_AdjustToasterOvenTemperature_visual \\
                   runs/exp1_AdjustWaterTemperature_visual
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

from contrastive_train import (
    DisentangleModel,
    PatchClipDataset,
)
from contrastive_train_subtask import (
    build_subtask_indices,
    load_instruction_meta,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_z(model: DisentangleModel, loader, device):
    model.eval()
    Z_t, Z_n, sids, cids, ep_idxs = [], [], [], [], []
    for batch in loader:
        patches, sid, cid, ep, _ = batch
        patches = patches.to(device, non_blocking=True)
        zt, zn = model(patches)
        Z_t.append(zt.cpu().numpy())
        Z_n.append(zn.cpu().numpy())
        sids.append(np.asarray(sid))
        cids.append(np.asarray(cid))
        ep_idxs.append(np.asarray(ep))
    return (
        np.concatenate(Z_t).astype(np.float32),
        np.concatenate(Z_n).astype(np.float32),
        np.concatenate(sids),
        np.concatenate(cids),
        np.concatenate(ep_idxs),
    )


def knn_predict(z_tr: np.ndarray, y_tr: np.ndarray, z_te: np.ndarray, k: int) -> np.ndarray:
    """L2-norm 가정. cosine sim top-k majority vote."""
    sim = z_te @ z_tr.T
    k = min(k, z_tr.shape[0])
    idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    nn_labels = y_tr[idx]
    # row-wise mode
    pred = np.empty(z_te.shape[0], dtype=y_tr.dtype)
    for i in range(z_te.shape[0]):
        c = Counter(nn_labels[i].tolist())
        pred[i] = c.most_common(1)[0][0]
    return pred


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_episode_mean(z: np.ndarray, ep_idx: np.ndarray):
    """Per-episode mean of z → re-normalize. Returns (M, D), (M,) ep_list."""
    eps = np.array(sorted(set(int(e) for e in ep_idx.tolist())))
    out = np.zeros((len(eps), z.shape[1]), dtype=np.float32)
    for i, e in enumerate(eps):
        m = ep_idx == e
        v = z[m].mean(0)
        n = np.linalg.norm(v)
        out[i] = v / max(n, 1e-8)
    return out, eps


def shorten(s: str, n: int = 32) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------------------
# A. Confusion matrix plot
# ---------------------------------------------------------------------------


def plot_confusion(
    cm: np.ndarray,
    labels: list[str],
    task_name: str,
    knn_k: int,
    accuracy: float,
    out_path: Path,
) -> None:
    """겹침 없는 confusion matrix.
    axes 의 라벨은 짧은 ID ("sub-task 0", "sub-task 1") 만.
    전체 instruction 은 figure caption 에.
    """
    n = cm.shape[0]
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)

    for i in range(n):
        for j in range(n):
            color = "white" if norm[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({norm[i, j]:.2f})",
                ha="center",
                va="center",
                fontsize=12,
                color=color,
            )

    short_ticks = [f"sub-task {i}" for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_ticks, fontsize=11)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_ticks, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(
        f"{task_name} — sub-task k-NN confusion (test, k={knn_k}, acc={accuracy:.3f})",
        fontsize=12,
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("row-normalized")

    # caption: 전체 instruction 매핑
    legend_text = "\n".join(f"sub-task {i}  =  {labels[i]}" for i in range(n))
    fig.text(
        0.5,
        0.02,
        legend_text,
        ha="center",
        va="bottom",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="none"),
    )

    plt.subplots_adjust(bottom=0.20 + 0.03 * n)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# D. Episode Gram plot
# ---------------------------------------------------------------------------


def plot_episode_gram(
    gram: np.ndarray,
    ep_subtasks_sorted: np.ndarray,
    subtask_labels: dict[int, str],
    task_name: str,
    stats: dict[str, float],
    out_path: Path,
) -> None:
    """겹침 없는 episode×episode Gram heatmap.

    레이아웃:
        ┌──┬──────────────┬─┐   상단:   sub-task 색띠 (어디까지 어느 sub-task)
        │  │   (top bar)  │ │
        ├──┼──────────────┼─┤
        │  │              │ │   메인:   heatmap (텍스트 없음)
        │L │   heatmap    │c │
        │e │              │b │
        │f │              │a │
        │t │              │r │
        ├──┼──────────────┼─┤
        │  │ (bottom: x)  │ │
        └──┴──────────────┴─┘
    + figure caption: 색 → sub-task 이름 매핑, within/cross 통계
    """
    from matplotlib.patches import Patch

    n = gram.shape[0]
    unique_st = sorted(set(int(x) for x in ep_subtasks_sorted))
    cmap_cat = plt.get_cmap("tab10")
    color_map = {st: cmap_cat(i) for i, st in enumerate(unique_st)}

    fig = plt.figure(figsize=(9, 8))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[0.025, 1.0, 0.04],
        height_ratios=[0.025, 1.0],
        wspace=0.02, hspace=0.02,
        left=0.06, right=0.92, top=0.92, bottom=0.30,
    )
    ax_top = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])

    # 메인 heatmap
    im = ax_main.imshow(gram, cmap="bwr", vmin=-1, vmax=1, aspect="auto")
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    # 블록 경계선
    boundaries = np.where(np.diff(ep_subtasks_sorted) != 0)[0]
    for b in boundaries:
        ax_main.axvline(b + 0.5, color="k", linewidth=1.0, alpha=0.7)
        ax_main.axhline(b + 0.5, color="k", linewidth=1.0, alpha=0.7)

    # 색 띠 — 상단 (가로)
    top_strip = np.array(
        [list(color_map[int(s)]) for s in ep_subtasks_sorted], dtype=float
    ).reshape(1, n, 4)
    ax_top.imshow(top_strip, aspect="auto")
    ax_top.set_xticks([])
    ax_top.set_yticks([])

    # 색 띠 — 좌측 (세로)
    left_strip = np.array(
        [list(color_map[int(s)]) for s in ep_subtasks_sorted], dtype=float
    ).reshape(n, 1, 4)
    ax_left.imshow(left_strip, aspect="auto")
    ax_left.set_xticks([])
    ax_left.set_yticks([])

    # Colorbar
    fig.colorbar(im, cax=ax_cbar, label="cosine similarity")

    # 제목
    fig.suptitle(
        f"{task_name} — episode × episode Gram of z_task (test set, sorted by sub-task)",
        fontsize=12,
        y=0.97,
    )

    # Caption: sub-task 색 → 라벨 + 통계
    legend_patches = [
        Patch(color=color_map[st], label=f"sub-task {st}  ({subtask_labels[st]})")
        for st in unique_st
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.13),
        ncol=1,
        frameon=False,
        fontsize=10,
    )

    stat_lines = [
        f"within sub-task cosine = {stats['within']:.4f}",
        f"cross  sub-task cosine = {stats['cross']:.4f}",
        f"delta  (within - cross) = {stats['within'] - stats['cross']:+.4f}   "
        f"(higher = better sub-task separation)",
        f"test episodes = {n}",
    ]
    fig.text(
        0.5,
        0.03,
        "\n".join(stat_lines),
        ha="center",
        va="bottom",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="none"),
    )

    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sample-level Gram (픽셀 단위)
# ---------------------------------------------------------------------------


def plot_sample_gram_pixel(
    Z_task: np.ndarray,
    Z_nuis: np.ndarray,
    sid: np.ndarray,
    cid: np.ndarray,
    ep_idx: np.ndarray,
    subtask_labels: dict[int, str],
    task_name: str,
    out_path: Path,
) -> None:
    """Test sample 전체의 z_task / z_nuis pairwise cosine matrix (1 sample = 1 픽셀).

    정렬 순서 (outer → inner):
        sub-task → episode → camera
    경계선:
        sub-task 경계만 굵게 (episode/camera 까지 그리면 너무 빽빽)
    """
    sort_key = np.lexsort((cid, ep_idx, sid))      # 가장 안쪽이 cid, outer 가 sid
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
        # 색 범위: 실제 값 분포 기반으로 자동 (대비 향상)
        # 대각선 (=1) 빼고 5~95 percentile 로
        off_diag = gram[~np.eye(n, dtype=bool)]
        vlo, vhi = np.percentile(off_diag, [2, 98])
        vrange = max(abs(vlo), abs(vhi))

        im = ax.imshow(
            gram,
            cmap="bwr",
            vmin=-vrange,
            vmax=vrange,
            interpolation="nearest",
            aspect="equal",
        )
        for b in boundaries_sub:
            ax.axvline(b + 0.5, color="k", linewidth=1.5, alpha=0.8)
            ax.axhline(b + 0.5, color="k", linewidth=1.5, alpha=0.8)

        ax.set_title(label, fontsize=14, fontweight="bold", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("cosine similarity", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        # sub-task 라벨 (블록 중앙, axes 외부)
        for st in sorted(subtask_labels.keys()):
            m = sid_s == st
            if not m.any():
                continue
            pos = float(np.where(m)[0].mean())
            ax.text(
                pos,
                -n * 0.025,
                f"sub-task {st}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    # figure 제목
    fig.suptitle(
        f"{task_name} — sample×sample cosine (n={n})   ·   sorted: sub-task → episode → camera",
        fontsize=12,
        y=0.98,
    )

    # figure-level caption (instruction 매핑)
    legend_text = "\n".join(
        f"sub-task {k}  =  {shorten(subtask_labels[k], 80)}"
        for k in sorted(subtask_labels.keys())
    )
    fig.text(
        0.5,
        0.02,
        legend_text,
        ha="center",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="none"),
    )

    plt.subplots_adjust(top=0.92, bottom=0.18, left=0.05, right=0.97, wspace=0.20)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Labeled pixel matrix (cell value + axis labels) — robocasa_*_heatmap.png 스타일
# ---------------------------------------------------------------------------


def plot_sample_gram_labeled(
    Z: np.ndarray,
    sid: np.ndarray,
    cid: np.ndarray,
    ep_idx: np.ndarray,
    cam_names: list[str],
    subtask_labels: dict[int, str],
    task_name: str,
    embed_name: str,
    out_path: Path,
) -> None:
    """모든 cell 에 값 표시 + 양 축에 sample 별 라벨 표시.

    참고: robocasa_start_hand_heatmap.png 스타일 (sample 마다 cell 안에 +0.6 등 값).
    n=528 기준: 이미지 약 12000×12000 px, ~150MB 이하.
    """
    sort_key = np.lexsort((cid, ep_idx, sid))
    Zs = Z[sort_key]
    sid_s = sid[sort_key]
    cid_s = cid[sort_key]
    ep_s = ep_idx[sort_key]

    gram = (Zs @ Zs.T).astype(np.float32)
    n = gram.shape[0]

    # Sample 라벨 — "ep007/s0/cam0" 같이 짧게
    cam_short = [c.replace("robot0_", "") for c in cam_names]
    labels = [
        f"ep{int(ep_s[i]):03d}/s{int(sid_s[i])}/{cam_short[int(cid_s[i])]}"
        for i in range(n)
    ]

    # Cell 당 픽셀 (작을수록 파일 작음, 너무 작으면 텍스트 깨짐)
    pix_per_cell = 22
    cell_font = 3
    label_font = 4

    # figsize 추정 — dpi=80 가정
    side_px = n * pix_per_cell + 1200  # margin
    side_in = side_px / 80.0
    fig, ax = plt.subplots(figsize=(side_in + 2, side_in))

    im = ax.imshow(
        gram, cmap="bwr", vmin=-1, vmax=1,
        interpolation="nearest", aspect="equal",
    )

    # Cell 값 (양/음 부호 표시, 1 소수점)
    for i in range(n):
        for j in range(n):
            v = gram[i, j]
            color = "white" if abs(v) > 0.55 else "black"
            ax.text(
                j, i, f"{v:+.1f}",
                ha="center", va="center",
                fontsize=cell_font, color=color,
            )

    # 축 라벨
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=label_font, family="monospace")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=label_font, family="monospace")

    # sub-task 블록 경계
    for b in np.where(np.diff(sid_s) != 0)[0]:
        ax.axvline(b + 0.5, color="k", linewidth=2.0)
        ax.axhline(b + 0.5, color="k", linewidth=2.0)

    # sub-task 라벨 (블록 중앙, axes 외부)
    for st in sorted(subtask_labels.keys()):
        m = sid_s == st
        if not m.any():
            continue
        pos = float(np.where(m)[0].mean())
        ax.text(
            pos, -n * 0.025,
            f"sub-task {st}: {shorten(subtask_labels[st], 50)}",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold",
        )

    ax.set_title(
        f"{task_name} — {embed_name}   sample × sample cosine (n={n})",
        fontsize=18, pad=24,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.015)
    cbar.set_label("cosine similarity", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# t-SNE 4-panel grid
# ---------------------------------------------------------------------------


def plot_tsne_grid(
    Zt: np.ndarray,
    Zn: np.ndarray,
    sid: np.ndarray,
    cid: np.ndarray,
    subtask_labels: dict[int, str],
    camera_names: list[str],
    task_name: str,
    out_path: Path,
    seed: int = 42,
) -> None:
    """test sample 전체를 t-SNE 로 2D 사영, 4 panel 동시 표시:

        ┌─ z_task colored by sub-task  ─┬─ z_task colored by camera   ─┐
        │  (분리되어야 좋음 ★)            │  (분리 안 되어야 좋음)         │
        ├─ z_nuis colored by sub-task  ─┼─ z_nuis colored by camera   ─┤
        │  (분리 안 되어야 좋음)           │  (분리되어야 좋음 ★)           │
        └────────────────────────────────┴────────────────────────────────┘
    """
    from sklearn.manifold import TSNE

    n = Zt.shape[0]
    perp = min(30, max(5, (n - 1) // 3))
    print(f"  [t-SNE] n_samples={n}, perplexity={perp}")
    Zt2 = TSNE(n_components=2, random_state=seed, perplexity=perp, init="pca").fit_transform(Zt)
    Zn2 = TSNE(n_components=2, random_state=seed, perplexity=perp, init="pca").fit_transform(Zn)

    cmap_sub = plt.get_cmap("tab10")
    cmap_cam = plt.get_cmap("Set1")

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    sub_keys = sorted(subtask_labels.keys())

    def panel(ax, Z2, labels, label_names, cmap, title, mark_emoji):
        for c, name in enumerate(label_names):
            m = labels == c
            if not m.any():
                continue
            ax.scatter(
                Z2[m, 0],
                Z2[m, 1],
                s=18,
                color=cmap(c),
                label=shorten(name, 38),
                alpha=0.7,
                edgecolors="none",
            )
        ax.set_title(f"{mark_emoji} {title}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=8, loc="best", framealpha=0.85, markerscale=1.2)

    # 4 panels
    panel(
        axes[0, 0], Zt2, sid,
        [subtask_labels[k] for k in sub_keys], cmap_sub,
        "z_task colored by SUB-TASK   (* should separate)", "[good?]",
    )
    panel(
        axes[0, 1], Zt2, cid, camera_names, cmap_cam,
        "z_task colored by CAMERA    (should not separate)", "[leak?]",
    )
    panel(
        axes[1, 0], Zn2, sid,
        [subtask_labels[k] for k in sub_keys], cmap_sub,
        "z_nuis colored by SUB-TASK  (should not separate)", "[leak?]",
    )
    panel(
        axes[1, 1], Zn2, cid, camera_names, cmap_cam,
        "z_nuis colored by CAMERA    (* should separate)", "[good?]",
    )

    fig.suptitle(
        f"{task_name} — t-SNE of test embeddings (n={n} samples)",
        fontsize=13,
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-run analysis
# ---------------------------------------------------------------------------


def analyze_run(run_dir: Path, ckpt_name: str, knn_k: int, device: str, do_labeled: bool = False) -> dict:
    out = run_dir / "analysis_exp1"
    out.mkdir(exist_ok=True)

    config = json.loads((run_dir / "config.json").read_text())
    task = config["task"]
    subtask_to_instr_str: dict[str, str] = config["subtask_to_instruction"]
    subtask_to_instr: dict[int, str] = {int(k): v for k, v in subtask_to_instr_str.items()}
    n_cls = len(subtask_to_instr)

    # 모델 재구성 — visual-only 와 VL run 둘 다 호환
    mc = config["model_config"]
    model = DisentangleModel(**mc).to(device)
    ckpt = torch.load(run_dir / ckpt_name, map_location=device, weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    elif "vl_model" in ckpt:
        vl_sd = ckpt["vl_model"]
        visual_sd = {
            k[len("visual."):]: v for k, v in vl_sd.items() if k.startswith("visual.")
        }
        model.load_state_dict(visual_sd)
    else:
        raise SystemExit(
            f"checkpoint 에 'model' / 'vl_model' key 없음: {list(ckpt.keys())}"
        )

    # 데이터 재구성 (같은 seed)
    instr_meta = load_instruction_meta(Path(config["instructions_json"]))
    seed = int(config["args"]["seed"])
    rng = np.random.default_rng(seed)
    train_idx, test_idx, _, cams = build_subtask_indices(
        Path(config["h5_path"]),
        instr_meta,
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

    # A. Confusion matrix
    pred_te = knn_predict(Zt_tr, sid_tr, Zt_te, k=knn_k)
    cm = confusion_matrix(sid_te, pred_te, n_cls)
    acc = float((pred_te == sid_te).mean())
    per_class_recall = []
    for c in range(n_cls):
        m = sid_te == c
        per_class_recall.append(float(((pred_te == c) & m).sum() / max(m.sum(), 1)))
    plot_confusion(
        cm,
        [subtask_to_instr[i] for i in range(n_cls)],
        task,
        knn_k,
        acc,
        out / "confusion_matrix.png",
    )

    # D. Episode × Episode Gram (test set)
    ep_to_subtask = {int(k): int(v) for k, v in instr_meta["episode_to_task_index"].items()}
    ep_mean, ep_list = per_episode_mean(Zt_te, ep_te)
    ep_subtasks = np.array([ep_to_subtask[int(e)] for e in ep_list])
    order = np.argsort(ep_subtasks, kind="stable")
    ep_mean_s = ep_mean[order]
    ep_subtasks_s = ep_subtasks[order]
    gram = ep_mean_s @ ep_mean_s.T

    # within / cross 분리 (Gram 의 상삼각만)
    within: list[float] = []
    cross: list[float] = []
    n_ep = gram.shape[0]
    for i in range(n_ep):
        for j in range(i + 1, n_ep):
            (within if ep_subtasks_s[i] == ep_subtasks_s[j] else cross).append(float(gram[i, j]))
    w_mean = float(np.mean(within)) if within else 0.0
    c_mean = float(np.mean(cross)) if cross else 0.0
    plot_episode_gram(
        gram,
        ep_subtasks_s,
        subtask_to_instr,
        task,
        {"within": w_mean, "cross": c_mean},
        out / "episode_gram.png",
    )

    # t-SNE 4-panel (sample 단위 = test 전체 점)
    plot_tsne_grid(
        Zt_te,
        Zn_te,
        sid_te,
        cid_te,
        subtask_to_instr,
        cams,
        task,
        out / "tsne_grid.png",
    )

    # Sample-level Gram (픽셀 단위, sub-task → episode → camera 정렬)
    plot_sample_gram_pixel(
        Zt_te,
        Zn_te,
        sid_te,
        cid_te,
        ep_te,
        subtask_to_instr,
        task,
        out / "sample_gram.png",
    )

    # Labeled (cell 값 + 축 라벨) — 큰 파일이라 옵션
    if do_labeled:
        print(f"  [labeled] z_task pixel-labeled gram rendering... (slow)")
        plot_sample_gram_labeled(
            Zt_te, sid_te, cid_te, ep_te, cams,
            subtask_to_instr, task, "z_task",
            out / "sample_gram_z_task_labeled.png",
        )
        print(f"  [labeled] z_nuis pixel-labeled gram rendering... (slow)")
        plot_sample_gram_labeled(
            Zn_te, sid_te, cid_te, ep_te, cams,
            subtask_to_instr, task, "z_nuis",
            out / "sample_gram_z_nuis_labeled.png",
        )

    majority = max(instr_meta["histogram"].values()) / instr_meta["num_episodes"]
    result = {
        "task": task,
        "num_episodes": int(instr_meta["num_episodes"]),
        "num_test_episodes": int(n_ep),
        "majority_class_rate": float(majority),
        "sub_tasks": subtask_to_instr,
        "knn_accuracy": acc,
        "per_class_recall": per_class_recall,
        "within_subtask_cos": w_mean,
        "cross_subtask_cos": c_mean,
        "within_minus_cross": w_mean - c_mean,
        "confusion_matrix": cm.tolist(),
    }
    (out / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    return result


# ---------------------------------------------------------------------------
# C. Summary plot (모든 task)
# ---------------------------------------------------------------------------


def plot_summary(summary_all: list[dict], out_path: Path) -> None:
    n = len(summary_all)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 왼쪽: k-NN acc vs majority
    ax = axes[0]
    x = np.arange(n)
    w = 0.35
    knn = [r["knn_accuracy"] for r in summary_all]
    maj = [r["majority_class_rate"] for r in summary_all]
    bars1 = ax.bar(x - w / 2, knn, w, label="sub-task k-NN acc", color="C0")
    bars2 = ax.bar(x + w / 2, maj, w, label="majority class", color="C1", alpha=0.8)
    for b, v in zip(bars1, knn):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    for b, v in zip(bars2, maj):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([r["task"] for r in summary_all], rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, label="chance (2-class)")
    ax.set_title("Exp 1a (visual-only): sub-task k-NN vs majority")
    ax.legend(loc="upper left")

    # 오른쪽: within / cross cosine
    ax = axes[1]
    w_in = [r["within_subtask_cos"] for r in summary_all]
    w_cr = [r["cross_subtask_cos"] for r in summary_all]
    bars1 = ax.bar(x - w / 2, w_in, w, label="within sub-task cos", color="C2")
    bars2 = ax.bar(x + w / 2, w_cr, w, label="cross sub-task cos", color="C3", alpha=0.8)
    for b, v in zip(bars1, w_in):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    for b, v in zip(bars2, w_cr):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([r["task"] for r in summary_all], rotation=15, ha="right")
    ax.set_ylabel("cosine similarity (per-episode mean z_task)")
    ax.set_title("Within vs Cross sub-task similarity (lower within−cross gap = harder)")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-dirs", nargs="+", required=True, type=Path)
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument(
        "--summary-out",
        type=Path,
        default=Path("runs/exp1_summary.json"),
    )
    p.add_argument("--device", default=None)
    p.add_argument(
        "--labeled",
        action="store_true",
        help="추가로 cell 값 + 축 라벨 표시된 거대한 sample Gram PNG 생성 (z_task/z_nuis 각각). 매우 느림 (~수 분).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}  knn_k={args.knn_k}")

    summary_all: list[dict] = []
    for rd in args.run_dirs:
        if not rd.exists():
            print(f"[skip] {rd} not found")
            continue
        print(f"\n=== analyzing {rd} ===")
        res = analyze_run(rd, args.ckpt, args.knn_k, device, do_labeled=args.labeled)
        summary_all.append(res)
        print(
            f"  task={res['task']}  knn_acc={res['knn_accuracy']:.3f}  "
            f"majority={res['majority_class_rate']:.3f}  "
            f"within={res['within_subtask_cos']:.3f} cross={res['cross_subtask_cos']:.3f}  "
            f"Δ={res['within_minus_cross']:+.3f}"
        )

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary_all, indent=2, ensure_ascii=False))
    plot_summary(summary_all, args.summary_out.with_suffix(".png"))

    print(f"\n[done] summary → {args.summary_out}")
    print(f"        plot    → {args.summary_out.with_suffix('.png')}")
    for r in summary_all:
        rd_name = f"runs/exp1_{r['task']}_visual"
        print(f"        {r['task']:>30}: {rd_name}/analysis_exp1/")


if __name__ == "__main__":
    main()
