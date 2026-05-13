"""두 LIBERO 소스(원본 HDF5 / LeRobot)의 contrastive 학습 결과를 side-by-side 비교.

run_libero_pipeline.py 의 끝에서 호출. index.json 의 sources 매핑을 읽어
양쪽 run dir 의 eval/ 산출물을 모은다.

산출:
    pair_dir/compare/
      gram_pair.png        2×3 (행=소스, 열=z_task / z_nuis / raw)
      umap_pair.png        2×2 (행=소스, 열=ztask-by-task / znuis-by-camera)
      metric_diff.json     eval_summary.json 의 numeric leaf 비교
      SUMMARY.md           사람이 읽는 요약

Bonus: 두 소스의 task centroid (10×D) Procrustes 정렬 후 cosine 상관 — 같은 task
geometry 를 유도하는지 정량화.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def flatten_metrics(d: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten_metrics(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def diff_metrics(a: dict, b: dict) -> dict[str, dict[str, float | None]]:
    """a (hdf5), b (lerobot) → {key: {hdf5, lerobot, diff}}"""
    flat_a = flatten_metrics(a)
    flat_b = flatten_metrics(b)
    keys = sorted(set(flat_a) | set(flat_b))
    out: dict[str, dict[str, float | None]] = {}
    for k in keys:
        va = flat_a.get(k)
        vb = flat_b.get(k)
        d = (vb - va) if (va is not None and vb is not None) else None
        out[k] = {"hdf5": va, "lerobot": vb, "diff": d}
    return out


def paste_panel(
    paths: list[list[Path]],
    titles_rows: list[str],
    titles_cols: list[str],
    out_path: Path,
    figsize_per: tuple[float, float] = (4.0, 4.0),
) -> None:
    """paths[i][j] 의 PNG 들을 행=row, 열=col 로 붙임. None 인 셀은 빈칸."""
    nr = len(paths)
    nc = max(len(r) for r in paths)
    fig, axes = plt.subplots(
        nr, nc, figsize=(nc * figsize_per[0], nr * figsize_per[1])
    )
    if nr == 1:
        axes = np.array([axes])
    if nc == 1:
        axes = axes[:, None]
    for i in range(nr):
        for j in range(nc):
            ax = axes[i, j]
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(titles_rows[i], fontsize=11, rotation=90,
                              labelpad=10)
            if i == 0 and j < len(titles_cols):
                ax.set_title(titles_cols[j], fontsize=11)
            if j < len(paths[i]) and paths[i][j] is not None and paths[i][j].exists():
                img = Image.open(paths[i][j])
                ax.imshow(np.asarray(img))
            else:
                ax.text(0.5, 0.5, "(missing)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def load_z_centroids(run_dir: Path) -> dict[str, np.ndarray] | None:
    """eval/test_embeddings.npz 가 있으면 task별 centroid (z_task) 반환."""
    npz = run_dir / "eval" / "test_embeddings.npz"
    if not npz.exists():
        return None
    z = np.load(npz)
    if "z_task" not in z or "task_ids" not in z:
        return None
    zt = z["z_task"]                # (N, D)
    tids = z["task_ids"]            # (N,)
    centroids: dict[int, np.ndarray] = {}
    for ti in np.unique(tids):
        mu = zt[tids == ti].mean(axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-9)
        centroids[int(ti)] = mu
    return centroids


def procrustes_corr(
    ca: dict[int, np.ndarray], cb: dict[int, np.ndarray]
) -> dict[str, float] | None:
    """두 centroid dict 의 Procrustes 정렬 후 평균 cosine + 행렬 상관."""
    common = sorted(set(ca) & set(cb))
    if len(common) < 3:
        return None
    A = np.stack([ca[k] for k in common], axis=0)  # (K, Da)
    B = np.stack([cb[k] for k in common], axis=0)  # (K, Db)
    # 차원 다르면 패딩
    Da, Db = A.shape[1], B.shape[1]
    if Da != Db:
        d = max(Da, Db)
        A = np.pad(A, ((0, 0), (0, d - Da)))
        B = np.pad(B, ((0, 0), (0, d - Db)))
    # Orthogonal Procrustes: min ||A R - B||  →  R = U V^T from SVD(A^T B)
    M = A.T @ B
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    A_rot = A @ R
    # row-wise cosine
    cos = (A_rot * B).sum(axis=1) / (
        np.linalg.norm(A_rot, axis=1) * np.linalg.norm(B, axis=1) + 1e-9
    )
    # gram 행렬 상관 (정렬 robust 확인용)
    G_a = A_rot @ A_rot.T
    G_b = B @ B.T
    g_corr = float(
        np.corrcoef(G_a.flatten(), G_b.flatten())[0, 1]
    )
    return {
        "n_tasks": len(common),
        "mean_cos": float(cos.mean()),
        "min_cos": float(cos.min()),
        "gram_corr": g_corr,
    }


def write_summary(
    pair_dir: Path,
    run_dirs: dict[str, Path],
    metric_diff: dict,
    procrustes: dict | None,
    smoke: bool,
) -> None:
    lines = [f"# LIBERO-goal pair comparison\n",
             f"- pair_dir: `{pair_dir}`",
             f"- smoke: **{smoke}**",
             f"- sources:"]
    for src, rd in run_dirs.items():
        cfg_path = rd / "config.json"
        n_tasks = "?"
        if cfg_path.exists():
            cfg = json.load(open(cfg_path))
            n_tasks = len(cfg.get("task_to_id", {}))
        lines.append(f"  - **{src}**: `{rd}`  (tasks={n_tasks})")
    lines.append("")

    lines.append("## Metric diff (lerobot − hdf5)\n")
    lines.append("| metric | hdf5 | lerobot | diff |")
    lines.append("|---|---:|---:|---:|")

    def fmt(x):
        if x is None:
            return "—"
        return f"{x:.4f}" if abs(x) < 1000 else f"{x:.2e}"

    for k, v in metric_diff.items():
        lines.append(
            f"| `{k}` | {fmt(v['hdf5'])} | {fmt(v['lerobot'])} | {fmt(v['diff'])} |"
        )
    lines.append("")

    if procrustes:
        lines.append("## Task centroid Procrustes alignment\n")
        lines.append(f"- common tasks: **{procrustes['n_tasks']}**")
        lines.append(f"- mean per-task cosine after rotation: **{procrustes['mean_cos']:.4f}**")
        lines.append(f"- min per-task cosine: **{procrustes['min_cos']:.4f}**")
        lines.append(f"- centroid Gram cross-correlation: **{procrustes['gram_corr']:.4f}**")
        lines.append("")
        lines.append("> mean_cos > 0.7 → 두 소스가 유사한 task geometry 를 유도. ")
        lines.append("> 낮으면 data-source artifacts (FPS, no-op 필터링 등) 가 영향.")
        lines.append("")
    else:
        lines.append("## Task centroid Procrustes alignment\n")
        lines.append("> test_embeddings.npz 없음 — eval 이 centroid 를 dump 하지 않음. ")
        lines.append("> contrastive_eval.py 에서 z_task / task_ids 저장 필요.\n")

    lines.append("## 패널\n")
    lines.append("- `gram_pair.png` — 2×3 (z_task / z_nuis / raw DINOv3) per 소스")
    lines.append("- `umap_pair.png` — 2×2 (ztask-by-task / znuis-by-camera) per 소스")
    lines.append("")
    lines.append("## 주의\n")
    lines.append("- Source A=20Hz, Source B=10Hz: clip_length=4 가 cover 하는 물리 시간 다름.")
    lines.append("- LeRobot port 는 no-op 필터링됨 → episode 수/길이 차이 있음.")
    lines.append("- 동일 task 만 비교한 정량 지표는 위 표 참고.")

    (pair_dir / "compare" / "SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair-dir", type=Path, required=True)
    args = p.parse_args()

    pair_dir = args.pair_dir
    index_path = pair_dir / "index.json"
    if not index_path.exists():
        raise SystemExit(f"index.json 없음: {index_path}")
    index = json.load(open(index_path))
    sources: dict[str, Path] = {k: Path(v) for k, v in index["sources"].items()}
    if "hdf5" not in sources or "lerobot" not in sources:
        raise SystemExit(f"hdf5/lerobot 모두 필요. got: {list(sources.keys())}")
    smoke = bool(index.get("smoke", False))

    out_dir = pair_dir / "compare"
    out_dir.mkdir(exist_ok=True)

    # metric diff
    a = json.load(open(sources["hdf5"] / "eval" / "eval_summary.json"))
    b = json.load(open(sources["lerobot"] / "eval" / "eval_summary.json"))
    md = diff_metrics(a, b)
    with open(out_dir / "metric_diff.json", "w") as f:
        json.dump(md, f, indent=2)

    # gram panel 2×3
    gram_paths = [
        [
            sources[src] / "eval" / f"gram_test_{name}.png"
            for name in ["ztask", "znuis", "dinov3_raw"]
        ]
        for src in ["hdf5", "lerobot"]
    ]
    paste_panel(
        gram_paths,
        titles_rows=["hdf5", "lerobot"],
        titles_cols=["z_task (test)", "z_nuis (test)", "DINOv3 raw (test)"],
        out_path=out_dir / "gram_pair.png",
        figsize_per=(5.0, 5.0),
    )

    # umap panel 2×2
    umap_paths = [
        [
            sources[src] / "eval" / "umap_ztask_by_task.png",
            sources[src] / "eval" / "umap_znuis_by_camera.png",
        ]
        for src in ["hdf5", "lerobot"]
    ]
    paste_panel(
        umap_paths,
        titles_rows=["hdf5", "lerobot"],
        titles_cols=["UMAP z_task by task", "UMAP z_nuis by camera"],
        out_path=out_dir / "umap_pair.png",
        figsize_per=(5.0, 5.0),
    )

    # procrustes
    ca = load_z_centroids(sources["hdf5"])
    cb = load_z_centroids(sources["lerobot"])
    procrustes = procrustes_corr(ca, cb) if (ca and cb) else None
    if procrustes:
        with open(out_dir / "procrustes.json", "w") as f:
            json.dump(procrustes, f, indent=2)

    write_summary(pair_dir, sources, md, procrustes, smoke)
    print(f"[done] compare → {out_dir}")


if __name__ == "__main__":
    main()
