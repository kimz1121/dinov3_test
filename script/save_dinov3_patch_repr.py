"""extract_robocasa_clips.py 가 저장한 PNG 들에 대해 DINOv3 patch token 추출.

save_dinov3_repr.py 의 mean-pool 임베딩과 달리 **spatial 구조 (H, W) 를 유지**한
(n, H, W, D) feature map 을 task 별 HDF5 파일에 저장한다. Contrastive 학습에서
attention pooling 의 입력 토큰으로 사용.

출력 구조:
    data/patch_embeddings/{Task}.hdf5
      attrs: model_id, embedder, sizes, dim, patch_h, patch_w, dtype,
             cameras, num_clips, clip_length, sampling_mode, seed,
             manifest_path, manifest_sha256
      data/
        demo_000/
          clip_0/
            robot0_agentview_left   (n, 14, 14, 384) fp16   attrs: frames_global, start_local
            robot0_agentview_right  ...
            robot0_eye_in_hand      ...
          clip_1/ ...
        demo_001/ ...

사용 예:
    # 전체 task, fp16, single-scale 224
    python script/save_dinov3_patch_repr.py

    # 빠른 검증용 (CheesyBread 만, batch 작게)
    python script/save_dinov3_patch_repr.py --tasks CheesyBread --batch-size 16

    # multi-scale opt-in
    python script/save_dinov3_patch_repr.py --sizes 448 224
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dinov3_embedders import DEFAULT_MODEL_ID, PatchSpatialEmbedder


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def png_path(
    png_root: Path, task: str, cam: str, ep_idx: int, clip_id: int, f_idx: int
) -> Path:
    return (
        png_root
        / task
        / cam
        / f"ep{ep_idx:03d}_clip{clip_id:02d}_f{f_idx:02d}.png"
    )


def encode_in_batches(
    embedder: PatchSpatialEmbedder, paths: list[Path], batch_size: int
) -> torch.Tensor:
    """PNG 경로 리스트 → (N, H, W, D) cpu float32 텐서.

    batch_size 만큼씩 PIL.open → encode_batch. encode_batch 가 cpu 로 내려주므로
    바로 concat.
    """
    outs: list[torch.Tensor] = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i : i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        feat = embedder.encode_batch(imgs)        # (B, H, W, D) float32
        outs.append(feat)
    return torch.cat(outs, dim=0) if outs else torch.zeros((0, 0, 0, 0))


def process_task(
    task: str,
    task_entry: dict,
    png_root: Path,
    out_path: Path,
    embedder: PatchSpatialEmbedder,
    batch_size: int,
    out_dtype: np.dtype,
    overwrite: bool,
    manifest_path: Path,
    manifest_sha256: str,
    sampling_mode: str,
    seed: int,
    num_clips: int,
    clip_length: int,
) -> tuple[int, int]:
    cameras: list[str] = task_entry["cameras"]
    episodes: list[dict] = task_entry["episodes"]

    # 이전 크래시로 남은 .tmp 가 있으면 청소 (skip 판정 전에 처리).
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        print(f"  [info] stale tmp 삭제: {tmp_path.name}")
        tmp_path.unlink()

    if out_path.exists() and not overwrite:
        print(f"  [skip] {out_path} exists (use --overwrite)")
        return 0, 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    H = W = embedder.max_size
    D = embedder.dim

    n_frames_written = 0
    n_clips_written = 0

    # Atomic write: tmp 에 다 쓰고 성공 시에만 진짜 이름으로 rename.
    # 중간에 죽으면 except 가 tmp 삭제 → skip-if-exists 가 다음에 정상 동작.
    try:
        with h5py.File(tmp_path, "w") as f:
            f.attrs["model_id"] = embedder.model_id
            f.attrs["embedder"] = embedder.name
            f.attrs["sizes"] = np.array(list(embedder.sizes), dtype=np.int32)
            f.attrs["dim"] = D
            f.attrs["patch_h"] = H
            f.attrs["patch_w"] = W
            f.attrs["dtype"] = str(np.dtype(out_dtype))
            f.attrs["cameras"] = np.array(cameras, dtype=h5py.string_dtype())
            f.attrs["num_clips"] = num_clips
            f.attrs["clip_length"] = clip_length
            f.attrs["sampling_mode"] = sampling_mode
            f.attrs["seed"] = seed
            f.attrs["manifest_path"] = str(manifest_path)
            f.attrs["manifest_sha256"] = manifest_sha256

            grp = f.create_group("data")

            for ep in tqdm(episodes, desc=f"  {task}", unit="ep"):
                ep_idx: int = ep["ep_idx"]
                demo_grp = grp.create_group(f"demo_{ep_idx:03d}")

                for clip in ep["clips"]:
                    clip_id: int = clip["clip_id"]
                    frames_global: list[int] = clip["frames_global"]
                    n = len(frames_global)
                    start_local: int = clip["start_local"]

                    clip_grp = demo_grp.create_group(f"clip_{clip_id}")

                    # (camera × n) 평탄화해 한 번에 encode
                    flat_paths: list[Path] = []
                    for cam in cameras:
                        for f_idx in range(n):
                            flat_paths.append(
                                png_path(png_root, task, cam, ep_idx, clip_id, f_idx)
                            )
                    feats = encode_in_batches(embedder, flat_paths, batch_size)
                    # (cam_count * n, H, W, D) → (cam, n, H, W, D)
                    feats = feats.reshape(len(cameras), n, H, W, D)
                    feats_np = feats.numpy().astype(out_dtype)

                    for ci, cam in enumerate(cameras):
                        dset = clip_grp.create_dataset(
                            cam,
                            data=feats_np[ci],
                            compression="gzip",
                            compression_opts=4,
                        )
                        dset.attrs["frames_global"] = np.array(
                            frames_global, dtype=np.int64
                        )
                        dset.attrs["start_local"] = start_local

                    n_clips_written += 1
                    n_frames_written += n * len(cameras)
        # 모든 쓰기 성공: tmp → 진짜 이름. POSIX rename 은 atomic.
        tmp_path.replace(out_path)
    except BaseException:
        # KeyboardInterrupt, OOM, h5py 에러 등 어떤 중단이든 partial tmp 청소.
        tmp_path.unlink(missing_ok=True)
        raise

    print(
        f"  saved → {out_path}  "
        f"({len(episodes)} demos, {n_clips_written} clips, {n_frames_written} frames)"
    )
    return n_clips_written, n_frames_written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--manifest", type=Path, default=Path("data/robocasa_clips/manifest.json")
    )
    p.add_argument("--png-root", type=Path, default=Path("data/robocasa_clips"))
    p.add_argument("--output-root", type=Path, default=Path("data/patch_embeddings"))
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[224],
        help="입력 정사각 해상도(픽셀) 1 개 이상. 기본 단일 스케일 224.",
    )
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--cameras", nargs="+", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
        help="HDF5 저장 dtype",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.manifest.exists():
        raise SystemExit(
            f"manifest 없음: {args.manifest}. 먼저 extract_robocasa_clips.py 실행"
        )
    manifest = load_manifest(args.manifest)
    manifest_sha = sha256_file(args.manifest)

    embedder = PatchSpatialEmbedder(
        model_id=args.model_id,
        device=args.device,
        sizes=tuple(args.sizes),
    )
    print(
        f"[INFO] embedder=patch_spatial dim={embedder.dim} "
        f"H=W={embedder.max_size} device={embedder.device} dtype_model={embedder.dtype}"
    )

    tasks = args.tasks or list(manifest["datasets"].keys())
    print(f"[INFO] tasks={tasks}")
    print(f"[INFO] manifest={args.manifest} sha256={manifest_sha[:12]}...")

    out_dtype = np.float16 if args.dtype == "float16" else np.float32

    total_clips = 0
    total_frames = 0
    for task in tasks:
        if task not in manifest["datasets"]:
            print(f"  [skip] {task} not in manifest")
            continue
        task_entry = manifest["datasets"][task]
        # cameras 필터 (옵션)
        if args.cameras is not None:
            allowed = set(args.cameras)
            task_entry = {
                **task_entry,
                "cameras": [c for c in task_entry["cameras"] if c in allowed],
            }

        out_path = args.output_root / f"{task}.hdf5"
        print(f"\n=== {task} → {out_path} ===")
        n_clips, n_frames = process_task(
            task=task,
            task_entry=task_entry,
            png_root=args.png_root,
            out_path=out_path,
            embedder=embedder,
            batch_size=args.batch_size,
            out_dtype=out_dtype,
            overwrite=args.overwrite,
            manifest_path=args.manifest,
            manifest_sha256=manifest_sha,
            sampling_mode=manifest["sampling_mode"],
            seed=manifest["seed"],
            num_clips=manifest["num_clips"],
            clip_length=manifest["clip_length"],
        )
        total_clips += n_clips
        total_frames += n_frames

    print(
        f"\n[done] total {total_clips} clips, {total_frames} frames over {len(tasks)} tasks"
    )


if __name__ == "__main__":
    main()
