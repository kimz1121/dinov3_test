"""학습된 disentanglement 모델 평가 — Gram heatmap, kNN, UMAP.

핵심 시각화는 **test-only Gram matrix**: 학습 중 본 적 없는 episode 만으로 cosine
similarity matrix 를 그려서 task 별 block-diagonal 패턴을 확인. raw DINOv3 (clip 의
mean-pool) baseline 과 비교해서 contrastive 학습의 효과를 대비.

출력:
    {out_dir}/
      gram_test_ztask.png         z_task 의 Gram (test only)
      gram_test_znuis.png         z_nuis 의 Gram (test only)
      gram_test_dinov3_raw.png    raw DINOv3 mean-pool 의 Gram (baseline)
      umap_ztask_by_task.png      z_task 색=task
      umap_ztask_by_camera.png    z_task 색=camera
      umap_znuis_by_camera.png    z_nuis 색=camera
      umap_znuis_by_task.png      z_nuis 색=task
      eval_summary.json           모든 수치

사용 예:
    python script/contrastive_eval.py --run-dir runs/20260512_141502
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from contrastive_train import (
    DisentangleModel,
    PatchClipDataset,
    SampleIndex,
    build_indices,
    discover_h5,
)
from similarity import plot_heatmap


CAMERA_ABBREV = {
    "robot0_agentview_left": "L",
    "robot0_agentview_right": "R",
    "robot0_eye_in_hand": "H",
}


def short_cam(cam: str) -> str:
    return CAMERA_ABBREV.get(cam, cam)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_z(
    model: DisentangleModel,
    loader: DataLoader,
    device: str,
) -> dict[str, np.ndarray]:
    """test/train loader 한 번 돌려서 z_task, z_nuis, label, id 모음."""
    zs_task, zs_nuis = [], []
    tasks, cams, eps, clips = [], [], [], []
    model.eval()
    for patches, task_id, cam_id, ep_idx, clip_id in loader:
        patches = patches.to(device, non_blocking=True)
        z_task, z_nuis = model(patches)
        zs_task.append(z_task.cpu().numpy())
        zs_nuis.append(z_nuis.cpu().numpy())
        tasks.append(task_id.numpy())
        cams.append(cam_id.numpy())
        eps.append(ep_idx.numpy())
        clips.append(clip_id.numpy())
    return {
        "z_task": np.concatenate(zs_task, axis=0),
        "z_nuis": np.concatenate(zs_nuis, axis=0),
        "task_id": np.concatenate(tasks, axis=0),
        "cam_id": np.concatenate(cams, axis=0),
        "ep_idx": np.concatenate(eps, axis=0),
        "clip_id": np.concatenate(clips, axis=0),
    }


def raw_dinov3_baseline(indices: list[SampleIndex]) -> np.ndarray:
    """(n, H, W, D) → spatial × temporal mean → L2 정규화 (clip 당 (D,)).

    HDF5 로부터 직접 읽어 model 통과 없이 baseline 임베딩 생성.
    """
    arrs = []
    h5_cache: dict[str, h5py.File] = {}
    for s in indices:
        if s.h5_path not in h5_cache:
            h5_cache[s.h5_path] = h5py.File(s.h5_path, "r", swmr=True)
        f = h5_cache[s.h5_path]
        arr = f[f"data/{s.demo_key}/{s.clip_key}/{s.camera}"][...]
        v = arr.astype(np.float32).mean(axis=(0, 1, 2))    # (D,)
        v = v / max(np.linalg.norm(v), 1e-8)
        arrs.append(v)
    for f in h5_cache.values():
        f.close()
    return np.stack(arrs, axis=0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def knn_accuracy(
    ref: np.ndarray, ref_labels: np.ndarray, q: np.ndarray, q_labels: np.ndarray, k: int
) -> float:
    """L2 정규화 가정 — cosine = dot. k-NN majority vote."""
    sim = q @ ref.T                                              # (Q, R)
    idx = np.argpartition(-sim, kth=min(k, ref.shape[0] - 1), axis=1)[:, :k]
    neigh = ref_labels[idx]                                      # (Q, k)
    # 다수결
    preds = np.array(
        [np.bincount(row).argmax() for row in neigh], dtype=q_labels.dtype
    )
    return float((preds == q_labels).mean())


def within_cross_cosine(
    z: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float]:
    """within-label / cross-label / separation 평균 cosine.

    self-pair (i==j) 는 제외.
    """
    G = z @ z.T                                                  # (N, N)
    same = labels[:, None] == labels[None, :]
    eye = np.eye(len(labels), dtype=bool)
    within_mask = same & ~eye
    cross_mask = ~same

    within = G[within_mask].mean() if within_mask.any() else 0.0
    cross = G[cross_mask].mean() if cross_mask.any() else 0.0
    return float(within), float(cross), float(within - cross)


# ---------------------------------------------------------------------------
# Gram heatmap helpers
# ---------------------------------------------------------------------------


def sort_by_task(
    indices: list[SampleIndex], task_to_id: dict[str, int]
) -> tuple[np.ndarray, list[int]]:
    """(task, camera, ep, clip) 순으로 안정정렬한 permutation 과 task block 경계.

    Returns:
        perm: indices 길이의 인덱스 배열
        boundaries: 각 task 시작 위치 (첫 task 제외, plot 의 boundaries 인자 용도)
    """
    keys = [
        (s.task_id, s.camera_id, s.ep_idx, s.clip_id) for s in indices
    ]
    perm = sorted(range(len(indices)), key=lambda i: keys[i])
    perm_arr = np.array(perm)

    sorted_tasks = [indices[i].task_id for i in perm]
    boundaries: list[int] = []
    for i in range(1, len(sorted_tasks)):
        if sorted_tasks[i] != sorted_tasks[i - 1]:
            boundaries.append(i)
    return perm_arr, boundaries


def make_labels(
    indices: list[SampleIndex], id_to_task: dict[int, str]
) -> list[str]:
    out = []
    for s in indices:
        out.append(
            f"{id_to_task[s.task_id]}/ep{s.ep_idx:03d}/c{s.clip_id}/{short_cam(s.camera)}"
        )
    return out


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------


def umap_plot(
    z: np.ndarray,
    color_labels: np.ndarray,
    color_names: list[str],
    title: str,
    out_path: Path,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> None:
    try:
        import umap   # umap-learn
    except ImportError as e:
        raise SystemExit(
            "umap-learn 필요. `pip install umap-learn` 후 다시 실행"
        ) from e
    import matplotlib.pyplot as plt

    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, max(2, len(z) - 1)),
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    emb2d = reducer.fit_transform(z)

    fig, ax = plt.subplots(figsize=(7, 6))
    n_cls = len(color_names)
    cmap = plt.get_cmap("tab10")
    for c in range(n_cls):
        m = color_labels == c
        if not m.any():
            continue
        ax.scatter(
            emb2d[m, 0],
            emb2d[m, 1],
            s=12,
            color=cmap(c % 10),
            label=color_names[c],
            alpha=0.7,
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="lower right", fontsize=9)
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
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument(
        "--patch-h5-root",
        type=Path,
        default=None,
        help="기본: config.json 의 값",
    )
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--umap-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--gram-cmap", default="bwr")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    if not run_dir.exists():
        raise SystemExit(f"run-dir not found: {run_dir}")

    out_dir = args.out_dir or run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json") as f:
        config = json.load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 데이터 ----
    patch_h5_root = args.patch_h5_root or Path(config["args"]["patch_h5_root"])
    tasks = config["args"].get("tasks")
    cameras_filter = config["args"].get("cameras")
    h5_paths = discover_h5(patch_h5_root, tasks)
    train_ratio = config["args"]["train_ratio"]

    train_idx, test_idx, task_to_id, cameras = build_indices(
        h5_paths, train_ratio, cameras_filter
    )
    id_to_task = {v: k for k, v in task_to_id.items()}

    train_ds = PatchClipDataset(train_idx)
    test_ds = PatchClipDataset(test_idx)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # ---- 모델 ----
    mc = config["model_config"]
    model = DisentangleModel(**mc).to(device)
    ckpt_path = run_dir / args.ckpt
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"[INFO] loaded {ckpt_path} (epoch={ckpt.get('epoch')})")

    # ---- z 모으기 ----
    print("[INFO] collecting z (train) ...")
    train_z = collect_z(model, train_loader, device)
    print("[INFO] collecting z (test) ...")
    test_z = collect_z(model, test_loader, device)

    # ---- Gram (test only) ----
    perm_test, task_boundaries = sort_by_task(test_idx, task_to_id)
    labels = make_labels([test_idx[i] for i in perm_test], id_to_task)
    n_test = len(perm_test)
    print(f"[INFO] test samples: {n_test}, task block boundaries={task_boundaries}")

    # z_task gram
    Z_t = test_z["z_task"][perm_test]
    G_t = Z_t @ Z_t.T
    plot_heatmap(
        G_t,
        labels,
        out_dir / "gram_test_ztask.png",
        title=f"Test-only Gram on z_task  (N={n_test})",
        cmap=args.gram_cmap,
        annotate=n_test <= 24,
        boundaries=task_boundaries,
    )

    # z_nuis gram (역시 task 정렬 — z_nuis 가 task 와 무관함을 보이려는 의도)
    Z_n = test_z["z_nuis"][perm_test]
    G_n = Z_n @ Z_n.T
    plot_heatmap(
        G_n,
        labels,
        out_dir / "gram_test_znuis.png",
        title=f"Test-only Gram on z_nuis  (N={n_test}, ordered by task)",
        cmap=args.gram_cmap,
        annotate=n_test <= 24,
        boundaries=task_boundaries,
    )

    # raw DINOv3 baseline
    print("[INFO] computing raw DINOv3 baseline ...")
    raw = raw_dinov3_baseline([test_idx[i] for i in perm_test])
    G_raw = raw @ raw.T
    plot_heatmap(
        G_raw,
        labels,
        out_dir / "gram_test_dinov3_raw.png",
        title=f"Test-only Gram on raw DINOv3 (clip mean-pool)  (N={n_test})",
        cmap=args.gram_cmap,
        annotate=n_test <= 24,
        boundaries=task_boundaries,
    )

    # ---- 정량 separation 지표 ----
    t_within, t_cross, t_sep = within_cross_cosine(
        test_z["z_task"], test_z["task_id"]
    )
    n_within, n_cross, n_sep = within_cross_cosine(
        test_z["z_nuis"], test_z["cam_id"]
    )
    # baseline (raw) — task separation 만 같이 비교
    raw_full = raw_dinov3_baseline(test_idx)
    r_within_t, r_cross_t, r_sep_t = within_cross_cosine(
        raw_full, test_z["task_id"]
    )

    # ---- kNN 정확도 (train→test) ----
    knn_task_on_ztask = knn_accuracy(
        train_z["z_task"], train_z["task_id"],
        test_z["z_task"], test_z["task_id"], args.knn_k,
    )
    knn_cam_on_ztask = knn_accuracy(
        train_z["z_task"], train_z["cam_id"],
        test_z["z_task"], test_z["cam_id"], args.knn_k,
    )
    knn_task_on_znuis = knn_accuracy(
        train_z["z_nuis"], train_z["task_id"],
        test_z["z_nuis"], test_z["task_id"], args.knn_k,
    )
    knn_cam_on_znuis = knn_accuracy(
        train_z["z_nuis"], train_z["cam_id"],
        test_z["z_nuis"], test_z["cam_id"], args.knn_k,
    )

    D1 = knn_task_on_ztask - knn_task_on_znuis
    D2 = knn_cam_on_znuis - knn_cam_on_ztask
    D_mean = (D1 + D2) / 2

    summary = {
        "ckpt": str(ckpt_path),
        "epoch": int(ckpt.get("epoch", -1)),
        "n_train_samples": int(len(train_idx)),
        "n_test_samples": int(len(test_idx)),
        "gram_separation": {
            "z_task_task": {
                "within": t_within,
                "cross": t_cross,
                "separation": t_sep,
            },
            "z_nuis_camera": {
                "within": n_within,
                "cross": n_cross,
                "separation": n_sep,
            },
            "raw_dinov3_task": {
                "within": r_within_t,
                "cross": r_cross_t,
                "separation": r_sep_t,
            },
        },
        "knn": {
            "k": args.knn_k,
            "task_on_ztask": knn_task_on_ztask,
            "cam_on_ztask": knn_cam_on_ztask,
            "task_on_znuis": knn_task_on_znuis,
            "cam_on_znuis": knn_cam_on_znuis,
        },
        "disentanglement": {
            "D1_task_zt_minus_zn": D1,
            "D2_cam_zn_minus_zt": D2,
            "D_mean": D_mean,
        },
    }
    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Eval summary ===")
    print(json.dumps(summary, indent=2))

    # ---- UMAP ----
    task_names = [id_to_task[i] for i in range(len(task_to_id))]
    cam_names = [short_cam(c) for c in cameras]

    print("\n[INFO] UMAP plots ...")
    umap_plot(
        test_z["z_task"], test_z["task_id"], task_names,
        "z_task colored by task (test)",
        out_dir / "umap_ztask_by_task.png",
        args.umap_neighbors, args.umap_min_dist, args.seed,
    )
    umap_plot(
        test_z["z_task"], test_z["cam_id"], cam_names,
        "z_task colored by camera (test)",
        out_dir / "umap_ztask_by_camera.png",
        args.umap_neighbors, args.umap_min_dist, args.seed,
    )
    umap_plot(
        test_z["z_nuis"], test_z["cam_id"], cam_names,
        "z_nuis colored by camera (test)",
        out_dir / "umap_znuis_by_camera.png",
        args.umap_neighbors, args.umap_min_dist, args.seed,
    )
    umap_plot(
        test_z["z_nuis"], test_z["task_id"], task_names,
        "z_nuis colored by task (test)",
        out_dir / "umap_znuis_by_task.png",
        args.umap_neighbors, args.umap_min_dist, args.seed,
    )

    print(f"\n[done] eval → {out_dir}")


if __name__ == "__main__":
    main()
