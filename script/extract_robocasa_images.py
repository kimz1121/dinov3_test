"""Robocasa LeRobot 데이터셋에서 시작/랜덤 프레임 PNG 추출.

HuggingFace에 올려둔 Robocasa LeRobot v3.0 데이터셋 3종에서,
에피소드마다 (1) 첫 프레임 (2) 랜덤 프레임을 카메라별로 PNG로 저장한다.

출력 구조:
    {output}/{start|random}/{데이터셋}/{카메라}/episode_{ep:03d}.png

사용 예:
    # 3개 데이터셋 전부 추출 (기본값)
    python script/extract_robocasa_images.py

    # 일부만, 시드/출력 변경
    python script/extract_robocasa_images.py \
        --datasets CheesyBread CloseCabinet \
        --output data/robocasa \
        --seed 42
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# torchvision의 video decoding deprecation 경고는 무시 (pyav 백엔드 경유 시 발생)
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# short name → HuggingFace repo id 매핑
_PREFIX = "kimz1121/robocasa_spatial_camrandom_images_pretrain_atomic_"
DATASETS: dict[str, str] = {
    "CheesyBread": _PREFIX + "CheesyBread",
    "CloseCabinet": _PREFIX + "CloseCabinet",
    "AdjustToasterOvenTemperature": _PREFIX + "AdjustToasterOvenTemperature",
    "CloseBlenderLid": _PREFIX + "CloseBlenderLid",
    "CloseDrawer": _PREFIX + "CloseDrawer",
    "CloseElectricKettleLid": _PREFIX + "CloseElectricKettleLid",
    "AdjustWaterTemperature": _PREFIX + "AdjustWaterTemperature",
    "CloseDishwasher": _PREFIX + "CloseDishwasher",
    "CloseFridge": _PREFIX + "CloseFridge",
    "CloseFridgeDrawer": _PREFIX + "CloseFridgeDrawer",
}

# torchcodec이 PyTorch 2.7과 ABI 비호환이라 pyav로 고정.
# (참고: pyav는 keyframe 단위 seek라 정확도는 약간 떨어지지만, 정지 프레임 한 장 추출엔 충분)
VIDEO_BACKEND = "pyav"


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """LeRobotDataset이 반환하는 (3, H, W) float32 [0, 1] 텐서를 PIL Image로 변환."""
    arr = image_tensor.detach().cpu().permute(1, 2, 0).numpy()  # (H, W, 3)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def short_camera_name(camera_key: str) -> str:
    """'observation.images.robot0_agentview_left' → 'robot0_agentview_left'."""
    prefix = "observation.images."
    return camera_key[len(prefix):] if camera_key.startswith(prefix) else camera_key


def extract_dataset(
    short_name: str,
    repo_id: str,
    output_root: Path,
    rng: np.random.Generator,
    exclude_start_from_random: bool,
    overwrite: bool,
) -> tuple[int, int]:
    """한 데이터셋에 대해 모든 에피소드의 start/random 프레임을 저장.

    Returns:
        (saved, skipped): 저장한 PNG 개수와 이미 존재하여 건너뛴 개수
    """
    print(f"\n[{short_name}] 로드 중 ({repo_id})...")
    ds = LeRobotDataset(repo_id, video_backend=VIDEO_BACKEND)
    camera_keys: list[str] = ds.meta.camera_keys
    print(
        f"  episodes={ds.num_episodes}, frames={ds.num_frames}, "
        f"fps={ds.fps}, cameras={len(camera_keys)}"
    )

    # 출력 폴더 미리 생성 — start/random × 카메라
    for kind in ("start", "random"):
        for cam in camera_keys:
            (output_root / kind / short_name / short_camera_name(cam)).mkdir(
                parents=True, exist_ok=True
            )

    saved = 0
    skipped = 0
    pbar = tqdm(range(ds.num_episodes), desc=f"  {short_name}", unit="ep")
    for ep_idx in pbar:
        row = ds.meta.episodes[ep_idx]
        start_global = int(row["dataset_from_index"])
        end_global = int(row["dataset_to_index"])  # exclusive

        # 랜덤 후보 범위: 시작 프레임을 제외할지 옵션
        low = start_global + 1 if exclude_start_from_random else start_global
        # 에피소드 길이가 1이면 (이론상 없겠지만) 시작 프레임 자체로 fallback
        if low >= end_global:
            random_global = start_global
        else:
            random_global = int(rng.integers(low=low, high=end_global))

        for kind, frame_idx in (("start", start_global), ("random", random_global)):
            # 모든 카메라가 이미 존재하면 디코딩 자체를 스킵하여 시간 절약
            target_paths = {
                cam: output_root
                / kind
                / short_name
                / short_camera_name(cam)
                / f"episode_{ep_idx:03d}.png"
                for cam in camera_keys
            }
            if not overwrite and all(p.exists() for p in target_paths.values()):
                skipped += len(target_paths)
                continue

            frame = ds[frame_idx]
            for cam, out_path in target_paths.items():
                if not overwrite and out_path.exists():
                    skipped += 1
                    continue
                tensor_to_pil(frame[cam]).save(out_path)
                saved += 1

    print(f"  saved={saved}, skipped(existing)={skipped}")
    return saved, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robocasa LeRobot 데이터셋에서 시작/랜덤 프레임 PNG 추출",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS.keys()),
        default=sorted(DATASETS.keys()),
        help="추출할 데이터셋 short name (여러 개 가능)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/robocasa"),
        help="출력 루트 디렉토리",
    )
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument(
        "--include-start-in-random",
        action="store_true",
        help="랜덤 프레임 후보에 첫 프레임(index 0)도 포함 (기본은 제외)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 존재하는 PNG를 덮어쓰기 (기본은 skip)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"출력 루트: {args.output.resolve()}")
    print(f"대상 데이터셋: {', '.join(args.datasets)}")
    print(f"seed={args.seed}, include_start_in_random={args.include_start_in_random}")

    rng = np.random.default_rng(args.seed)

    total_saved = 0
    total_skipped = 0
    for short_name in args.datasets:
        saved, skipped = extract_dataset(
            short_name=short_name,
            repo_id=DATASETS[short_name],
            output_root=args.output,
            rng=rng,
            exclude_start_from_random=not args.include_start_in_random,
            overwrite=args.overwrite,
        )
        total_saved += saved
        total_skipped += skipped

    print(
        f"\n완료. 총 saved={total_saved}, skipped(existing)={total_skipped}, "
        f"출력 폴더: {args.output}"
    )


if __name__ == "__main__":
    main()
