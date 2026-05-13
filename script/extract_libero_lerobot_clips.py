"""HuggingFaceVLA/libero (LeRobot v2) 에서 libero_goal task 들의 clip PNG 추출.

LIBERO LeRobot 포팅본은 1 repo 에 4 suite × ~10 task 가 task_index 로 구분된다.
이 스크립트는:
  1) suite 별 task_index 범위 (libero_goal 은 10..19) 로 task 필터
  2) task 별 episode 수집 → task-local ep_idx 부여
  3) extract_robocasa_clips 의 sample_clip_starts 로 clip 위치 결정
  4) LeRobotDataset[frame_global] 로 디코딩해서 PNG 저장
  5) manifest.json 작성 (RoboCasa/HDF5 와 동일 스키마)

카메라 매핑:
    observation.images.image  → agentview
    observation.images.image2 → eye_in_hand

사용 예:
    python script/extract_libero_lerobot_clips.py
    python script/extract_libero_lerobot_clips.py --tasks put_the_bowl_on_the_plate turn_on_the_stove --max-episodes 5
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from extract_robocasa_clips import episode_sub_rng, sample_clip_starts
from extract_robocasa_images import VIDEO_BACKEND, tensor_to_pil


REPO_ID = "HuggingFaceVLA/libero"

# 4 suite 의 task_index 범위. libero_goal=10..19, libero_spatial=20..29 등 (확인된 범위).
SUITE_TASK_INDEX = {
    "libero_spatial": (0, 9),
    "libero_object": (20, 29),
    "libero_goal": (10, 19),
    "libero_10": (None, None),  # 명시되지 않음 — task_index 범위는 verify 필요
}

CAM_MAP = {
    "observation.images.image": "agentview",
    "observation.images.image2": "eye_in_hand",
}
FPS = 10.0  # LeRobot port 는 10Hz (no-op 필터링됨)


def description_to_task_name(desc: str) -> str:
    """'put the bowl on the plate' → 'put_the_bowl_on_the_plate'."""
    return desc.strip().lower().replace(" ", "_")


def build_task_map(repo_id: str, suite: str) -> dict[int, str]:
    """task_index → snake_case task name. suite 의 범위로 필터.

    LeRobotDataset 객체의 ds.meta.tasks 는 버전에 따라 dict 또는 DataFrame 으로
    나오는데 형식이 불안정해서 (DataFrame.items() 가 컬럼명 yield 함 등),
    parquet 을 직접 받아서 처리.
    """
    path = hf_hub_download(repo_id, "meta/tasks.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    # 두 가지 가능한 레이아웃 모두 지원:
    #   (a) index=description(str),    column 'task_index' (int)
    #   (b) index=task_index(int),    column 'task' (str) 또는 첫 column
    if "task_index" in df.columns and df.index.dtype == object:
        items = [(int(v), str(k)) for k, v in df["task_index"].items()]
    else:
        df2 = df.reset_index()
        # task_index 가 컬럼에 있고 description 컬럼명을 찾는다
        ti_col = "task_index" if "task_index" in df2.columns else df2.columns[0]
        desc_col = next(c for c in df2.columns if c != ti_col)
        items = [(int(r[ti_col]), str(r[desc_col])) for _, r in df2.iterrows()]

    lo, hi = SUITE_TASK_INDEX.get(suite, (None, None))
    out: dict[int, str] = {}
    for ti, desc in items:
        if lo is not None and not (lo <= ti <= hi):
            continue
        out[ti] = description_to_task_name(desc)
    if not out:
        raise SystemExit(
            f"suite={suite} 의 task_index 가 tasks.parquet 와 매칭 안 됨. "
            f"sample items: {items[:5]}"
        )
    return out


def collect_episodes_from_parquet(
    repo_id: str, task_index_set: set[int]
) -> dict[int, list[tuple[int, int, int]]]:
    """meta/episodes/*.parquet 직접 읽어서 task_index → [(ep, from, to), ...] 생성.

    ds.hf_dataset 직접 인덱싱은 video decode 트리거 + 메타 컬럼명도 달라질 수 있어
    parquet 으로 우회. stats/task_index/mean 은 episode 의 task_index 와 동일
    (모든 frame 이 같은 task_index).
    """
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")
    ep_files = sorted([f for f in files if f.startswith("meta/episodes/") and f.endswith(".parquet")])

    rows: list[dict] = []
    for f in ep_files:
        p = hf_hub_download(repo_id, f, repo_type="dataset")
        df = pd.read_parquet(p, columns=[
            "episode_index", "dataset_from_index", "dataset_to_index",
            "length", "stats/task_index/mean",
        ])
        rows.append(df)
    eps = pd.concat(rows, ignore_index=True)
    eps["task_index"] = eps["stats/task_index/mean"].astype(int)

    by_task: dict[int, list[tuple[int, int, int]]] = {}
    for _, r in eps.iterrows():
        ti = int(r["task_index"])
        if ti in task_index_set:
            by_task.setdefault(ti, []).append((
                int(r["episode_index"]),
                int(r["dataset_from_index"]),
                int(r["dataset_to_index"]),
            ))
    # 안정성: ep_idx 오름차순 정렬
    for ti in by_task:
        by_task[ti].sort(key=lambda x: x[0])
    return by_task


def extract_one_task(
    ds: LeRobotDataset,
    task: str,
    episodes: list[tuple[int, int, int]],
    output_root: Path,
    rng_seed: int,
    n: int,
    m: int,
    mode: str,
    max_episodes: int | None,
    overwrite: bool,
) -> tuple[dict, int, int]:
    short_cams = list(CAM_MAP.values())
    for sc in short_cams:
        (output_root / task / sc).mkdir(parents=True, exist_ok=True)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    episodes_entries: list[dict] = []
    saved = 0
    skipped = 0

    print(f"  [{task}] {len(episodes)} episodes")
    pbar = tqdm(episodes, desc=f"  {task}", unit="ep", leave=False)
    for ep_local, (global_ep, f_from, f_to) in enumerate(pbar):
        length = f_to - f_from
        rng = episode_sub_rng(rng_seed, task, ep_local)
        starts_local = sample_clip_starts(length, n=n, m=m, mode=mode, rng=rng)
        clips: list[dict] = []
        for clip_id, sl in enumerate(starts_local):
            clips.append(
                {
                    "clip_id": clip_id,
                    "start_local": int(sl),
                    "frames_global": [f_from + sl + k for k in range(n)],
                }
            )
        episodes_entries.append(
            {
                "ep_idx": ep_local,
                "global_ep_idx": int(global_ep),
                "dataset_from_index": int(f_from),
                "dataset_to_index": int(f_to),
                "length": int(length),
                "clips": clips,
            }
        )

        # frame → (clip_id, f_idx) 매핑
        usages: dict[int, list[tuple[int, int]]] = {}
        for clip in clips:
            for f_idx, fg in enumerate(clip["frames_global"]):
                usages.setdefault(fg, []).append((clip["clip_id"], f_idx))

        # 기존 PNG 가 다 있는 frame 은 디코딩 스킵
        need_decode: list[int] = []
        for fg, uses in usages.items():
            for clip_id, f_idx in uses:
                for sc in short_cams:
                    out_path = (
                        output_root
                        / task
                        / sc
                        / f"ep{ep_local:03d}_clip{clip_id:02d}_f{f_idx:02d}.png"
                    )
                    if overwrite or not out_path.exists():
                        if fg not in need_decode:
                            need_decode.append(fg)
                        break

        for fg in need_decode:
            frame = ds[fg]
            for clip_id, f_idx in usages[fg]:
                for long_name, sc in CAM_MAP.items():
                    out_path = (
                        output_root
                        / task
                        / sc
                        / f"ep{ep_local:03d}_clip{clip_id:02d}_f{f_idx:02d}.png"
                    )
                    if not overwrite and out_path.exists():
                        skipped += 1
                        continue
                    tensor_to_pil(frame[long_name]).save(out_path)
                    saved += 1

        for fg, uses in usages.items():
            if fg in need_decode:
                continue
            skipped += len(uses) * len(short_cams)

    return (
        {
            "repo_id": REPO_ID,
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
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--suite", default="libero_goal",
                   choices=list(SUITE_TASK_INDEX.keys()))
    p.add_argument("--tasks", nargs="+", default=None,
                   help="task 이름 필터 (snake_case 형식)")
    p.add_argument("--output", type=Path, default=Path("data/libero_goal_lerobot_clips"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-clips", type=int, default=8)
    p.add_argument("--clip-length", type=int, default=4)
    p.add_argument("--sampling-mode", choices=["uniform", "random"], default="uniform")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="task 당 episode cap (smoke 용)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # 메타 (task map + episode→task) 는 parquet 직접 사용. ds.meta.tasks 가
    # 버전별로 dict/DataFrame 가변이라 우회.
    task_map = build_task_map(args.repo_id, args.suite)
    print(f"[INFO] suite={args.suite} → {len(task_map)} tasks: "
          f"{sorted(task_map.values())}")

    tasks_filter = set(args.tasks) if args.tasks else None
    task_index_set = {
        ti for ti, t in task_map.items()
        if tasks_filter is None or t in tasks_filter
    }
    if not task_index_set:
        raise SystemExit(f"no tasks matched filter {tasks_filter}")

    print(f"[INFO] episodes 메타 parquet 읽는 중...")
    by_task = collect_episodes_from_parquet(args.repo_id, task_index_set)

    print(f"[INFO] LeRobotDataset({args.repo_id}) 로드 중 (video decode 백엔드)...")
    ds = LeRobotDataset(args.repo_id, video_backend=VIDEO_BACKEND)
    print(f"  num_episodes={ds.num_episodes}, num_frames={ds.num_frames}, fps={ds.fps}")
    for ti in sorted(task_index_set):
        n = len(by_task.get(ti, []))
        print(f"  task_index={ti:>2} ({task_map[ti]:50s}) eps={n}")

    print(f"출력 루트: {args.output.resolve()}")
    manifest: dict = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "libero_lerobot",
        "suite": args.suite,
        "repo_id": args.repo_id,
        "seed": args.seed,
        "sampling_mode": args.sampling_mode,
        "num_clips": args.num_clips,
        "clip_length": args.clip_length,
        "datasets": {},
    }

    total_saved = 0
    total_skipped = 0
    for ti in sorted(task_index_set):
        task = task_map[ti]
        eps = by_task.get(ti, [])
        if not eps:
            print(f"  [skip] {task} (no episodes)")
            continue
        entry, saved, skipped = extract_one_task(
            ds=ds,
            task=task,
            episodes=eps,
            output_root=args.output,
            rng_seed=args.seed,
            n=args.clip_length,
            m=args.num_clips,
            mode=args.sampling_mode,
            max_episodes=args.max_episodes,
            overwrite=args.overwrite,
        )
        entry["task_index"] = int(ti)
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
