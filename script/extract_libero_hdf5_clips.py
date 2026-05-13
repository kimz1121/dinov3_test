"""원본 LIBERO HDF5 (yifengzhu-hf/LIBERO-datasets 미러) 에서 clip PNG 추출.

extract_robocasa_clips.py 의 LIBERO 버전. LeRobot 대신 native HDF5
(`data/demo_N/obs/{agentview_rgb, eye_in_hand_rgb}`, 256×256 uint8) 를 직접 읽는다.

샘플링 (episode_sub_rng, sample_clip_starts) 은 extract_robocasa_clips 에서 import
해서 재사용 → 같은 (seed, task, ep) 에서 동일 frame 선택.

출력 구조 (extract_robocasa_clips 와 동일):
    {output}/
      manifest.json
      {task}/
        {camera}/
          ep{ep:03d}_clip{c:02d}_f{f:02d}.png

사용 예:
    # libero_goal 전 task 추출
    python script/extract_libero_hdf5_clips.py

    # smoke
    python script/extract_libero_hdf5_clips.py \
        --tasks put_the_bowl_on_the_plate turn_on_the_stove \
        --max-episodes 5
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm

from extract_robocasa_clips import episode_sub_rng, sample_clip_starts


REPO_ID = "yifengzhu-hf/LIBERO-datasets"
CAM_MAP = {
    "agentview_rgb": "agentview",
    "eye_in_hand_rgb": "eye_in_hand",
}
FPS = 20.0  # LIBERO native sim FPS


def download_suite(suite: str, cache_dir: Path | None) -> Path:
    """libero_goal HDF5 셋을 HF 에서 받아 로컬 디렉토리 경로 반환."""
    print(f"[INFO] snapshot_download({REPO_ID}, {suite}/*.hdf5) ...")
    local = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=f"{suite}/*.hdf5",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    suite_dir = Path(local) / suite
    if not suite_dir.exists():
        raise SystemExit(f"suite dir not found after download: {suite_dir}")
    return suite_dir


def task_name_from_filename(p: Path) -> str:
    """open_the_middle_drawer_of_the_cabinet_demo.hdf5 → open_the_middle_drawer_of_the_cabinet."""
    name = p.stem
    if name.endswith("_demo"):
        name = name[: -len("_demo")]
    return name


def extract_one_task(
    h5_path: Path,
    task: str,
    output_root: Path,
    rng_seed: int,
    n: int,
    m: int,
    mode: str,
    max_episodes: int | None,
    overwrite: bool,
) -> tuple[dict, int, int]:
    short_cams = list(CAM_MAP.values())  # ["agentview", "eye_in_hand"]
    for cam in short_cams:
        (output_root / task / cam).mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    episodes_entries: list[dict] = []

    with h5py.File(h5_path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        if max_episodes is not None:
            demos = demos[:max_episodes]
        print(f"  [{task}] {len(demos)} episodes")
        pbar = tqdm(demos, desc=f"  {task}", unit="ep", leave=False)
        for ep_local, demo_key in enumerate(pbar):
            ep_grp = f[f"data/{demo_key}/obs"]
            T = ep_grp["agentview_rgb"].shape[0]
            rng = episode_sub_rng(rng_seed, task, ep_local)
            starts_local = sample_clip_starts(T, n=n, m=m, mode=mode, rng=rng)
            clips: list[dict] = []
            for clip_id, sl in enumerate(starts_local):
                clips.append(
                    {
                        "clip_id": clip_id,
                        "start_local": int(sl),
                        # 원본 HDF5 는 episode-local. global index 가 없어서 local 을 그대로 기록.
                        "frames_global": [int(sl + k) for k in range(n)],
                    }
                )
            episodes_entries.append(
                {
                    "ep_idx": ep_local,
                    "demo_key": demo_key,
                    "length": int(T),
                    "clips": clips,
                }
            )

            # 어느 frame 을 읽어야 하나
            need: dict[int, list[tuple[int, int]]] = {}
            for clip in clips:
                for f_idx, fl in enumerate(clip["frames_global"]):
                    need.setdefault(fl, []).append((clip["clip_id"], f_idx))

            # 이미 모든 PNG 가 있으면 디코딩 스킵
            need_read = []
            for fl, uses in need.items():
                for clip_id, f_idx in uses:
                    for sc in short_cams:
                        out_path = (
                            output_root
                            / task
                            / sc
                            / f"ep{ep_local:03d}_clip{clip_id:02d}_f{f_idx:02d}.png"
                        )
                        if overwrite or not out_path.exists():
                            if fl not in need_read:
                                need_read.append(fl)
                            break

            for fl in need_read:
                # batch read: 두 카메라 한 번씩
                imgs = {
                    sc: ep_grp[long_name][fl]  # (256,256,3) uint8
                    for long_name, sc in CAM_MAP.items()
                }
                for clip_id, f_idx in need[fl]:
                    for sc in short_cams:
                        out_path = (
                            output_root
                            / task
                            / sc
                            / f"ep{ep_local:03d}_clip{clip_id:02d}_f{f_idx:02d}.png"
                        )
                        if not overwrite and out_path.exists():
                            skipped += 1
                            continue
                        Image.fromarray(imgs[sc]).save(out_path)
                        saved += 1

            # 디코딩 스킵된 frame 의 PNG 카운트
            for fl, uses in need.items():
                if fl in need_read:
                    continue
                skipped += len(uses) * len(short_cams)

    return (
        {
            "repo_id": f"{REPO_ID}::{h5_path.name}",
            "fps": FPS,
            "cameras": short_cams,
            "episodes": episodes_entries,
        },
        saved,
        skipped,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="필터링할 task 이름 리스트 (기본: suite 내 전체)")
    p.add_argument("--output", type=Path, default=Path("data/libero_goal_hdf5_clips"))
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="HuggingFace 캐시 디렉토리 override")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-clips", type=int, default=8)
    p.add_argument("--clip-length", type=int, default=4)
    p.add_argument("--sampling-mode", choices=["uniform", "random"], default="uniform")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="task 당 episode 수 cap (smoke 용)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    suite_dir = download_suite(args.suite, args.cache_dir)
    h5_files = sorted(suite_dir.glob("*.hdf5"))
    tasks_filter = set(args.tasks) if args.tasks else None
    filtered: list[tuple[str, Path]] = []
    for p in h5_files:
        t = task_name_from_filename(p)
        if tasks_filter is None or t in tasks_filter:
            filtered.append((t, p))
    if not filtered:
        raise SystemExit(f"no tasks matched in {suite_dir}")

    print(f"출력 루트: {args.output.resolve()}")
    print(
        f"대상 ({len(filtered)} tasks): {', '.join(t for t,_ in filtered)} "
        f"| seed={args.seed} | m={args.num_clips} n={args.clip_length} "
        f"mode={args.sampling_mode} | max_episodes={args.max_episodes}"
    )

    manifest: dict = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "libero_hdf5",
        "suite": args.suite,
        "seed": args.seed,
        "sampling_mode": args.sampling_mode,
        "num_clips": args.num_clips,
        "clip_length": args.clip_length,
        "datasets": {},
    }

    total_saved = 0
    total_skipped = 0
    for task, h5_path in filtered:
        entry, saved, skipped = extract_one_task(
            h5_path=h5_path,
            task=task,
            output_root=args.output,
            rng_seed=args.seed,
            n=args.clip_length,
            m=args.num_clips,
            mode=args.sampling_mode,
            max_episodes=args.max_episodes,
            overwrite=args.overwrite,
        )
        manifest["datasets"][task] = entry
        total_saved += saved
        total_skipped += skipped
        print(f"  [{task}] saved={saved} skipped={skipped}")

    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        backup = manifest_path.with_suffix(
            f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        manifest_path.rename(backup)
        print(f"[INFO] 기존 manifest → {backup}")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[done] saved={total_saved} skipped={total_skipped}")
    print(f"manifest → {manifest_path}")


if __name__ == "__main__":
    main()
