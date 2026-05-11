"""Robocasa PNG 폴더 트리에서 DINOv3 임베딩을 추출해 HDF5 로 저장.

Lotus `save_dinov2_repr.py` 를 DINOv3 + 우리 데이터 레이아웃에 맞게 적응:

입력
    data/robocasa/
      {kind}/                      # start | random
        {Task}/                    # CheesyBread, CloseCabinet, ...
          {camera}/                # robot0_agentview_left, _right, _eye_in_hand
            episode_{ep:03d}.png

출력
    data/embeddings/dinov3_{embedder}/{kind}/{Task}.hdf5
      data/
        demo_000/
          robot0_agentview_left   shape (D,) float32
          robot0_agentview_right  shape (D,)
          robot0_eye_in_hand      shape (D,)
        demo_001/ ...
      attrs: model_id, embedder, num_episodes, cameras, dim

사용 예
    python script/save_dinov3_repr.py --embedder cls --tasks CheesyBread --kinds start
    python script/save_dinov3_repr.py --embedder patch_mean
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from dinov3_embedders import DEFAULT_MODEL_ID, build_embedder


def discover_tasks(kind_dir: Path) -> list[str]:
    return sorted(p.name for p in kind_dir.iterdir() if p.is_dir())


def discover_cameras(task_dir: Path) -> list[str]:
    return sorted(p.name for p in task_dir.iterdir() if p.is_dir())


def list_episode_pngs(camera_dir: Path) -> list[Path]:
    return sorted(camera_dir.glob("episode_*.png"))


def encode_camera(
    embedder, png_paths: list[Path], batch_size: int
) -> np.ndarray:
    """카메라 하나의 모든 episode PNG → (N, D) numpy float32."""
    out: list[np.ndarray] = []
    for i in range(0, len(png_paths), batch_size):
        batch_paths = png_paths[i : i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        emb = embedder.encode_batch(imgs)         # (B, D) cpu float32
        out.append(emb.numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, embedder.dim), np.float32)


def process_kind_task(
    embedder,
    input_root: Path,
    output_root: Path,
    kind: str,
    task: str,
    cameras: list[str] | None,
    batch_size: int,
    overwrite: bool,
) -> tuple[int, int]:
    """(kind, task) 하나에 대해 모든 카메라 처리 → 1개 HDF5 출력.

    Returns (num_episodes, num_cameras).
    """
    task_dir = input_root / kind / task
    cams = cameras or discover_cameras(task_dir)
    if not cams:
        print(f"  [skip] no cameras in {task_dir}")
        return 0, 0

    # episode 인덱스는 모든 카메라에서 동일해야 자연스럽지만,
    # 만약 카메라마다 다르다면 합집합을 쓰고 누락 처리.
    cam_pngs: dict[str, list[Path]] = {c: list_episode_pngs(task_dir / c) for c in cams}
    num_eps_per_cam = {c: len(v) for c, v in cam_pngs.items()}
    if len(set(num_eps_per_cam.values())) != 1:
        print(f"  [warn] camera episode counts differ: {num_eps_per_cam}")
    num_episodes = max(num_eps_per_cam.values()) if num_eps_per_cam else 0

    out_path = output_root / f"dinov3_{embedder.name}" / kind / f"{task}.hdf5"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        print(f"  [skip] {out_path} exists (use --overwrite)")
        return num_episodes, len(cams)

    # 카메라별로 인코딩 (메모리 절약: 한 카메라 끝나면 텐서 해제)
    cam_embeddings: dict[str, np.ndarray] = {}
    for cam in cams:
        paths = cam_pngs[cam]
        if not paths:
            print(f"  [warn] {cam}: no episodes")
            cam_embeddings[cam] = np.zeros((0, embedder.dim), np.float32)
            continue
        print(f"  encoding {cam} ({len(paths)} frames)...")
        cam_embeddings[cam] = encode_camera(embedder, paths, batch_size)

    # HDF5 쓰기 — reference 의 demo_{i} 구조를 모방하되 camera 를 dataset 으로
    with h5py.File(out_path, "w") as f:
        f.attrs["model_id"] = embedder.model_id
        f.attrs["embedder"] = embedder.name
        f.attrs["num_episodes"] = num_episodes
        f.attrs["cameras"] = np.array(cams, dtype=h5py.string_dtype())
        f.attrs["dim"] = embedder.dim

        grp = f.create_group("data")
        for ep in range(num_episodes):
            demo_grp = grp.create_group(f"demo_{ep:03d}")
            for cam in cams:
                arr = cam_embeddings[cam]
                if ep < len(arr):
                    demo_grp.create_dataset(cam, data=arr[ep].astype(np.float32))
                # else: 누락된 episode → dataset 생성하지 않음

    print(f"  saved → {out_path}  ({num_episodes} demos × {len(cams)} cameras × {embedder.dim}d)")
    return num_episodes, len(cams)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--embedder",
        choices=["cls", "patch_mean"],
        default="cls",
        help="pooling 방식: cls(빠름) | patch_mean(reference 스타일, multi-scale)",
    )
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--input-root", type=Path, default=Path("data/robocasa"))
    p.add_argument("--output-root", type=Path, default=Path("data/embeddings"))
    p.add_argument(
        "--kinds",
        nargs="+",
        default=["start", "random"],
        help="처리할 kind 디렉토리들",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="처리할 task 이름들 (기본: input-root 에서 자동 검출)",
    )
    p.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="카메라 이름 화이트리스트 (기본: 자동 검출)",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 HDF5 가 있어도 덮어쓰기",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[INFO] embedder={args.embedder}  model={args.model_id}")
    print(f"[INFO] input={args.input_root}  output={args.output_root}")

    embedder = build_embedder(args.embedder, model_id=args.model_id)
    print(f"[INFO] dim={embedder.dim}  device={embedder.device}  dtype={embedder.dtype}")

    total_demos = 0
    total_files = 0
    for kind in args.kinds:
        kind_dir = args.input_root / kind
        if not kind_dir.is_dir():
            print(f"[skip] {kind_dir} not found")
            continue

        tasks = args.tasks or discover_tasks(kind_dir)
        for task in tasks:
            print(f"\n=== {kind} / {task} ===")
            n_ep, n_cam = process_kind_task(
                embedder=embedder,
                input_root=args.input_root,
                output_root=args.output_root,
                kind=kind,
                task=task,
                cameras=args.cameras,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
            )
            total_demos += n_ep
            total_files += 1

    print(f"\n[done] wrote {total_files} HDF5 file(s), {total_demos} total demos")


if __name__ == "__main__":
    main()
