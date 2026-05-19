"""Robocasa contrastive 학습 결과에서 구분 잘된/못된 사례 이미지 시각화.

구분 못된 케이스: 서로 다른 task 인데 z_task cosine similarity 가 높은 쌍
구분 잘된 케이스: 서로 다른 task 인데 z_task cosine similarity 가 낮은 쌍

출력:
    {out_dir}/hard_confused_cases.png   — 구분 못된 사례 (cross-task high-sim)
    {out_dir}/easy_separated_cases.png  — 구분 잘된 사례 (cross-task low-sim)
    {out_dir}/within_task_cases.png     — 같은 task 내부 (참고용)

사용 예:
    python script/visualize_hard_cases.py --run-dir runs/robocasa_6task_full_20260512_052054
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from contrastive_train import (
    DisentangleModel,
    PatchClipDataset,
    SampleIndex,
    build_indices,
    discover_h5,
)

# patch_h5_root 이름 → 클립 이미지 폴더 매핑
_H5ROOT_TO_CLIPS: dict[str, str] = {
    "patch_embeddings":                       "robocasa_clips",
    "patch_embeddings_last":                  "robocasa_clips",
    "patch_embeddings_libero_goal_hdf5":      "libero_goal_hdf5_clips",
    "patch_embeddings_libero_goal_hdf5_last": "libero_goal_hdf5_clips",
    "patch_embeddings_libero_goal_lerobot":   "libero_goal_lerobot_clips",
    "patch_embeddings_libero_goal_lerobot_last": "libero_goal_lerobot_clips",
}

# 카메라 우선순위 — 시각화에 쓸 카메라 선택
PREFERRED_CAMS = ["robot0_agentview_left", "agentview"]


def clips_root_from_h5root(patch_h5_root: str | Path) -> Path:
    """patch_h5_root 경로에서 클립 이미지 폴더를 자동 판별."""
    stem = Path(patch_h5_root).name
    folder = _H5ROOT_TO_CLIPS.get(stem)
    if folder is None:
        raise SystemExit(f"clips 폴더 매핑 없음: {stem!r}  — _H5ROOT_TO_CLIPS 에 추가 필요")
    return Path("data") / folder


def build_original_clip_id_cache(h5_paths: list[Path]) -> dict[tuple, int]:
    """HDF5 clip attrs 에서 original_clip_id 를 미리 읽어 캐시.

    Returns: {(h5_path_str, demo_key, clip_key): original_clip_id}
    Last-clip 전용 HDF5 (clip_0 하나만 있고 attrs 에 original_clip_id 기록)에서만
    의미 있음. 일반 HDF5 는 clip key 번호 자체가 곧 original id 이므로 저장 안 함.
    """
    cache: dict[tuple, int] = {}
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as f:
            for demo_key in f["data"].keys():
                for clip_key in f[f"data/{demo_key}"].keys():
                    oid = f[f"data/{demo_key}/{clip_key}"].attrs.get("original_clip_id")
                    if oid is not None:
                        cache[(str(h5_path), demo_key, clip_key)] = int(oid)
    return cache


def get_clip_image(
    clips_root: Path,
    task: str,
    camera: str,
    ep_idx: int,
    clip_id: int,
    frame: int = 0,
) -> Image.Image | None:
    """clips_root/{task}/{camera}/ep{ep}_clip{clip}_f{frame}.png 로드."""
    cam_dir = clips_root / task / camera
    filename = f"ep{ep_idx:03d}_clip{clip_id:02d}_f{frame:02d}.png"
    img_path = cam_dir / filename
    if img_path.exists():
        return Image.open(img_path).convert("RGB")
    return None


def pick_camera(cameras: list[str]) -> str:
    for pref in PREFERRED_CAMS:
        if pref in cameras:
            return pref
    return cameras[0]


@torch.no_grad()
def collect_z_task(
    model: DisentangleModel,
    loader: DataLoader,
    device: str,
) -> np.ndarray:
    model.eval()
    zs = []
    for patches, *_ in loader:
        patches = patches.to(device, non_blocking=True)
        z_task, _ = model(patches)
        zs.append(z_task.cpu().numpy())
    return np.concatenate(zs, axis=0)


def find_pairs(
    z: np.ndarray,
    task_ids: np.ndarray,
    indices: list[SampleIndex],
    n_top: int = 8,
) -> tuple[list[tuple], list[tuple]]:
    """cross-task 쌍 중 cosine similarity 상위/하위 n_top 개 반환.

    Returns:
        confused: [(sim, i, j), ...] 높은 순 (구분 못한 케이스)
        separated: [(sim, i, j), ...] 낮은 순 (구분 잘 한 케이스)
    """
    sim_matrix = z @ z.T                           # (N, N), L2-norm 가정
    N = len(z)

    cross_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if task_ids[i] != task_ids[j]:
                cross_pairs.append((float(sim_matrix[i, j]), i, j))

    cross_pairs.sort(key=lambda x: -x[0])
    confused = cross_pairs[:n_top]                 # 가장 헷갈린 쌍

    cross_pairs.sort(key=lambda x: x[0])
    separated = cross_pairs[:n_top]               # 가장 잘 구분된 쌍

    return confused, separated


def render_pair_grid(
    pairs: list[tuple],
    indices: list[SampleIndex],
    id_to_task: dict[int, str],
    cameras: list[str],
    title: str,
    out_path: Path,
    clips_root: Path,
    orig_clip_cache: dict[tuple, int],
    pairs_per_row: int = 2,
) -> None:
    """각 쌍을 배경 박스 + vs 라벨로 묶어 compact grid 로 시각화."""
    import matplotlib.patches as mpatches

    def display_clip_id(s: SampleIndex) -> int:
        """last-clip HDF5 는 original_clip_id 로 이미지 경로 결정."""
        return orig_clip_cache.get((s.h5_path, s.demo_key, s.clip_key), s.clip_id)

    cam = pick_camera(cameras)
    n = len(pairs)
    n_rows = (n + pairs_per_row - 1) // pairs_per_row

    # 열 배치: (img, vs, img) per pair, 쌍 사이 separator
    col_widths = []
    for p in range(pairs_per_row):
        col_widths += [3.0, 0.45, 3.0]
        if p < pairs_per_row - 1:
            col_widths.append(0.5)

    fig_w = sum(col_widths) * 0.95
    fig_h = n_rows * 2.8

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)

    gs = gridspec.GridSpec(
        n_rows, len(col_widths),
        figure=fig,
        width_ratios=col_widths,
        hspace=0.55,
        wspace=0.05,
    )

    def col_for_pair(p: int):
        base = p * 4
        return base, base + 1, base + 2

    # ── 1단계: axes 생성 + 이미지 그리기 ──
    pair_axes: list[tuple] = []

    for idx, (sim, i, j) in enumerate(pairs):
        row = idx // pairs_per_row
        p   = idx % pairs_per_row

        si, sj   = indices[i], indices[j]
        task_i   = id_to_task[si.task_id]
        task_j   = id_to_task[sj.task_id]
        cid_i    = display_clip_id(si)
        cid_j    = display_clip_id(sj)
        img_i    = get_clip_image(clips_root, task_i, cam, si.ep_idx, cid_i)
        img_j    = get_clip_image(clips_root, task_j, cam, sj.ep_idx, cid_j)

        sim_color = "#d62728" if sim > 0.5 else "#1f77b4"
        box_color = "#fff0f0" if sim > 0.5 else "#f0f4ff"
        col_l, col_vs, col_r = col_for_pair(p)

        def make_img_ax(col, img, s, task, cid):
            ax = fig.add_subplot(gs[row, col])
            ax.axis("on")
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "not found", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
            short = task if len(task) <= 20 else task[:18] + ".."
            ax.set_xlabel(f"{short}\nep{s.ep_idx:03d} clip{cid:02d}",
                          fontsize=7, labelpad=2)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(sim_color); sp.set_linewidth(1.8)
            return ax

        ax_l = make_img_ax(col_l, img_i, si, task_i, cid_i)
        ax_r = make_img_ax(col_r, img_j, sj, task_j, cid_j)

        ax_vs = fig.add_subplot(gs[row, col_vs])
        ax_vs.axis("off")
        ax_vs.text(0.5, 0.62, "vs", ha="center", va="center",
                   fontsize=9, fontweight="bold", color=sim_color,
                   transform=ax_vs.transAxes)
        ax_vs.text(0.5, 0.35, f"{sim:.3f}", ha="center", va="center",
                   fontsize=8, color=sim_color, fontweight="bold",
                   transform=ax_vs.transAxes)

        pair_axes.append((sim_color, box_color, ax_l, ax_r))

    # ── 2단계: 한 번 draw → tightbbox 확정 → 박스 그리기 ──
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    pad = 4  # display pixel 패딩

    for sim_color, box_color, ax_l, ax_r in pair_axes:
        bb_l = ax_l.get_tightbbox(renderer)
        bb_r = ax_r.get_tightbbox(renderer)
        if bb_l is None or bb_r is None:
            continue
        # display → figure 좌표
        x0, y0 = inv.transform((min(bb_l.x0, bb_r.x0) - pad,
                                 min(bb_l.y0, bb_r.y0) - pad))
        x1, y1 = inv.transform((max(bb_l.x1, bb_r.x1) + pad,
                                 max(bb_l.y1, bb_r.y1) + pad))
        rect = mpatches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.008",
            linewidth=1.5,
            edgecolor=sim_color,
            facecolor=box_color,
            alpha=0.45,
            zorder=0,
            transform=fig.transFigure,
            clip_on=False,
        )
        fig.add_artist(rect)

    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dirs", type=Path, nargs="+", required=True,
                   help="하나 이상의 run 디렉토리 (각각 독립 처리)")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--n-cases", type=int, default=8, help="confused / separated 각 n 개")
    p.add_argument("--pairs-per-row", type=int, default=2, help="한 행에 배치할 쌍 수")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_one(run_dir: Path, args: argparse.Namespace, device: str) -> None:
    out_dir = run_dir / "hard_cases"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json") as f:
        config = json.load(f)

    patch_h5_root = Path(config["args"]["patch_h5_root"])
    tasks_filter = config["args"].get("tasks")
    cameras_filter = config["args"].get("cameras")
    h5_paths = discover_h5(patch_h5_root, tasks_filter)
    train_ratio = config["args"]["train_ratio"]

    clips_root = clips_root_from_h5root(patch_h5_root)
    print(f"[INFO] clips_root = {clips_root}")

    print("[INFO] building original_clip_id cache ...")
    orig_clip_cache = build_original_clip_id_cache(h5_paths)
    if orig_clip_cache:
        print(f"[INFO] original_clip_id cache: {len(orig_clip_cache)} entries (last-clip dataset)")
    else:
        print("[INFO] no original_clip_id attrs found (full-clip dataset)")

    _, test_idx, task_to_id, cameras = build_indices(h5_paths, train_ratio, cameras_filter)
    id_to_task = {v: k for k, v in task_to_id.items()}

    test_loader = DataLoader(
        PatchClipDataset(test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    mc = config["model_config"]
    model = DisentangleModel(**mc).to(device)
    ckpt_path = run_dir / args.ckpt
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"[INFO] loaded {ckpt_path}  (epoch={ckpt.get('epoch')})")

    print("[INFO] collecting z_task ...")
    z_task = collect_z_task(model, test_loader, device)
    task_ids = np.array([s.task_id for s in test_idx])
    print(f"[INFO] z_task shape={z_task.shape}")

    confused, separated = find_pairs(z_task, task_ids, test_idx, n_top=args.n_cases)

    tag = run_dir.name
    render_pair_grid(
        confused, test_idx, id_to_task, cameras,
        f"[{tag}]  Confused — different tasks, HIGH z_task sim",
        out_dir / "hard_confused_cases.png",
        clips_root=clips_root,
        orig_clip_cache=orig_clip_cache,
        pairs_per_row=args.pairs_per_row,
    )
    render_pair_grid(
        separated, test_idx, id_to_task, cameras,
        f"[{tag}]  Well-Separated — different tasks, LOW z_task sim",
        out_dir / "easy_separated_cases.png",
        clips_root=clips_root,
        orig_clip_cache=orig_clip_cache,
        pairs_per_row=args.pairs_per_row,
    )
    print(f"[done] → {out_dir}\n")


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for run_dir in args.run_dirs:
        if not run_dir.exists():
            print(f"[SKIP] not found: {run_dir}")
            continue
        print(f"\n{'='*60}")
        print(f"  Processing: {run_dir}")
        print(f"{'='*60}")
        run_one(run_dir, args, device)


if __name__ == "__main__":
    main()
