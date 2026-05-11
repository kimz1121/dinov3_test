"""Robocasa LeRobot 데이터셋에서 episode 당 m 개의 clip (각 n 프레임) PNG 추출.

extract_robocasa_images.py 의 start/random 1 장 추출 패턴을 확장한 버전.
Contrastive learning 용으로 episode 내에서 m 개의 clip 위치를 잡고, 각 clip 마다
n 개 연속 프레임을 카메라별로 PNG 로 저장한다. 동시에 sampling 정보 (frame 인덱스
등) 를 manifest.json 에 기록해서 임베딩 추출 단계에서 재현 가능하게 한다.

출력 구조:
    {output}/
      manifest.json
      {Task}/
        {camera}/
          ep{ep:03d}_clip{c:02d}_f{f:02d}.png

manifest.json 스키마는 docstring 하단 참고.

사용 예:
    # 기본값 (n=4, m=8, uniform sampling) 으로 3 task 전부
    python script/extract_robocasa_clips.py

    # CheesyBread 만, random sampling
    python script/extract_robocasa_clips.py \
        --datasets CheesyBread --sampling-mode random --seed 7

manifest.json (요약):
    {
      "seed": 42, "sampling_mode": "uniform",
      "num_clips": 8, "clip_length": 4,
      "datasets": {
        "CheesyBread": {
          "repo_id": "...",
          "fps": 20,
          "cameras": ["robot0_agentview_left", ...],
          "episodes": [
            {"ep_idx": 0, "length": 234,
             "clips": [
               {"clip_id": 0, "start_local": 0,
                "frames_global": [0, 1, 2, 3]}, ...
             ]}
          ]
        }
      }
    }
"""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from extract_robocasa_images import (
    DATASETS,
    VIDEO_BACKEND,
    short_camera_name,
    tensor_to_pil,
)


def episode_sub_rng(seed: int, task: str, ep_idx: int) -> np.random.Generator:
    """(seed, task, ep_idx) 에서 결정적으로 sub-rng 생성.

    task 를 추가/제거해도 다른 task 의 sampling 이 안 흔들리도록 task 이름을 시드에
    섞는다. hash() 는 PYTHONHASHSEED 의존이라 sha256 으로 안정화.
    """
    payload = f"{seed}|{task}|{ep_idx}".encode("utf-8")
    sub = int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**32)
    return np.random.default_rng(sub)


def sample_clip_starts(
    length: int, n: int, m: int, mode: str, rng: np.random.Generator
) -> list[int]:
    """episode 길이 length 에서 길이 n 짜리 clip 의 시작 인덱스 m 개 반환.

    valid_starts = length - n + 1 이 m 보다 작으면 가능한 만큼만 반환.
    """
    valid_starts = length - n + 1
    if valid_starts <= 0:
        return []
    m_eff = min(m, valid_starts)

    if mode == "uniform":
        # linspace 로 균등 분포 → 정수화 → dedup (짧은 episode 에서 충돌 가능)
        if m_eff == 1:
            return [0]
        raw = np.floor(np.linspace(0, valid_starts - 1, m_eff)).astype(int)
        seen: set[int] = set()
        starts: list[int] = []
        for s in raw:
            if int(s) not in seen:
                seen.add(int(s))
                starts.append(int(s))
        # dedup 후 모자라면 뒤쪽부터 채움
        s = valid_starts - 1
        while len(starts) < m_eff and s >= 0:
            if s not in seen:
                starts.append(s)
                seen.add(s)
            s -= 1
        return sorted(starts)

    if mode == "random":
        choice = rng.choice(valid_starts, size=m_eff, replace=False)
        return sorted(int(x) for x in choice)

    raise ValueError(f"unknown sampling mode: {mode}")


def build_episode_entry(
    ds: LeRobotDataset,
    ep_idx: int,
    n: int,
    m: int,
    mode: str,
    rng: np.random.Generator,
) -> dict:
    row = ds.meta.episodes[ep_idx]
    start_global = int(row["dataset_from_index"])
    end_global = int(row["dataset_to_index"])  # exclusive
    length = end_global - start_global

    starts_local = sample_clip_starts(length, n, m, mode, rng)
    clips = []
    for clip_id, sl in enumerate(starts_local):
        frames_global = [start_global + sl + k for k in range(n)]
        clips.append(
            {"clip_id": clip_id, "start_local": int(sl), "frames_global": frames_global}
        )
    return {
        "ep_idx": ep_idx,
        "dataset_from_index": start_global,
        "dataset_to_index": end_global,
        "length": length,
        "clips": clips,
    }


def extract_dataset(
    short_name: str,
    repo_id: str,
    output_root: Path,
    rng_seed: int,
    n: int,
    m: int,
    mode: str,
    overwrite: bool,
) -> tuple[dict, int, int]:
    """한 데이터셋 처리: PNG 저장 + episode/clip entry 리스트 반환.

    Returns:
        (dataset_manifest_entry, saved, skipped)
    """
    print(f"\n[{short_name}] 로드 중 ({repo_id})...")
    ds = LeRobotDataset(repo_id, video_backend=VIDEO_BACKEND)
    camera_keys: list[str] = ds.meta.camera_keys
    short_cams = [short_camera_name(c) for c in camera_keys]
    print(
        f"  episodes={ds.num_episodes}, frames={ds.num_frames}, "
        f"fps={ds.fps}, cameras={len(camera_keys)}"
    )

    # 카메라별 출력 폴더 생성
    for cam in short_cams:
        (output_root / short_name / cam).mkdir(parents=True, exist_ok=True)

    # 1) 모든 episode 의 clip 계획부터 만들고 manifest 결정
    episodes_entries: list[dict] = []
    for ep_idx in range(ds.num_episodes):
        rng = episode_sub_rng(rng_seed, short_name, ep_idx)
        episodes_entries.append(
            build_episode_entry(ds, ep_idx, n=n, m=m, mode=mode, rng=rng)
        )

    saved = 0
    skipped = 0

    # 2) 실제 디코딩 — episode 마다 사용 frame 만 dedup decode
    pbar = tqdm(episodes_entries, desc=f"  {short_name}", unit="ep")
    for ep_entry in pbar:
        ep_idx = ep_entry["ep_idx"]

        # (frame_global, clip_id, frame_in_clip) 매핑 — 같은 frame 이 여러 clip 에
        # 등장할 수 있게 (clip overlap) 일반화
        usages: dict[int, list[tuple[int, int]]] = {}
        for clip in ep_entry["clips"]:
            for f_idx, fg in enumerate(clip["frames_global"]):
                usages.setdefault(fg, []).append((clip["clip_id"], f_idx))

        # 어떤 frame 을 디코딩해야 하는지 결정 — 이미 전부 존재하면 디코딩 스킵
        need_decode: list[int] = []
        for fg, uses in usages.items():
            for clip_id, f_idx in uses:
                for short_cam in short_cams:
                    out_path = (
                        output_root
                        / short_name
                        / short_cam
                        / f"ep{ep_idx:03d}_clip{clip_id:02d}_f{f_idx:02d}.png"
                    )
                    if overwrite or not out_path.exists():
                        if fg not in need_decode:
                            need_decode.append(fg)
                        break  # 이 frame 은 어쨌든 디코딩 필요

        # 디코딩 + 저장
        for fg in need_decode:
            frame = ds[fg]
            for clip_id, f_idx in usages[fg]:
                for cam, short_cam in zip(camera_keys, short_cams):
                    out_path = (
                        output_root
                        / short_name
                        / short_cam
                        / f"ep{ep_idx:03d}_clip{clip_id:02d}_f{f_idx:02d}.png"
                    )
                    if not overwrite and out_path.exists():
                        skipped += 1
                        continue
                    tensor_to_pil(frame[cam]).save(out_path)
                    saved += 1

        # 디코딩 안 한 frame 의 PNG 가 이미 있으면 skipped 카운트
        for fg, uses in usages.items():
            if fg in need_decode:
                continue
            skipped += len(uses) * len(camera_keys)

    print(f"  saved={saved}, skipped(existing)={skipped}")

    return (
        {
            "repo_id": repo_id,
            "fps": float(ds.fps),
            "cameras": short_cams,
            "episodes": episodes_entries,
        },
        saved,
        skipped,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS.keys()),
        default=sorted(DATASETS.keys()),
    )
    p.add_argument("--output", type=Path, default=Path("data/robocasa_clips"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-clips", type=int, default=8, help="episode 당 clip 개수 m")
    p.add_argument("--clip-length", type=int, default=4, help="clip 당 frame 수 n")
    p.add_argument(
        "--sampling-mode",
        choices=["uniform", "random"],
        default="uniform",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"

    print(f"출력 루트: {args.output.resolve()}")
    print(
        f"대상: {', '.join(args.datasets)}  | seed={args.seed} "
        f"| m={args.num_clips} n={args.clip_length} mode={args.sampling_mode}"
    )

    manifest: dict = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "sampling_mode": args.sampling_mode,
        "num_clips": args.num_clips,
        "clip_length": args.clip_length,
        "datasets": {},
    }

    total_saved = 0
    total_skipped = 0
    for short_name in args.datasets:
        entry, saved, skipped = extract_dataset(
            short_name=short_name,
            repo_id=DATASETS[short_name],
            output_root=args.output,
            rng_seed=args.seed,
            n=args.clip_length,
            m=args.num_clips,
            mode=args.sampling_mode,
            overwrite=args.overwrite,
        )
        manifest["datasets"][short_name] = entry
        total_saved += saved
        total_skipped += skipped

    if manifest_path.exists() and not args.overwrite:
        # 기존 manifest 와 충돌 가능성 — 일단 백업 후 덮어쓰기
        backup = manifest_path.with_suffix(
            f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        manifest_path.rename(backup)
        print(f"[INFO] 기존 manifest → {backup} 로 백업")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[done] saved={total_saved}, skipped={total_skipped}")
    print(f"manifest → {manifest_path}")


if __name__ == "__main__":
    main()
