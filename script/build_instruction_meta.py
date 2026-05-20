"""LeRobot dataset 의 tasks.parquet + episodes.parquet 를 읽어서
patch_embeddings 옆에 사이드카 JSON 으로 instruction 매핑을 만든다.

출력 구조:
    data/patch_embeddings/{Task}_instructions.json
    {
      "task": "CloseDrawer",
      "dataset_repo": "kimz1121/...",
      "num_episodes": 110,
      "instructions": {"0": "Close the left drawer.", "1": "Close the right drawer."},
      "episode_to_task_index": {"0": 0, "1": 1, "2": 0, ...},
      "histogram": {"Close the left drawer.": 59, "Close the right drawer.": 51}
    }

LeRobot cache 우선 순위:
    1. ~/.cache/huggingface/lerobot/<repo>/meta/...
    2. ~/.cache/huggingface/hub/datasets--<repo>/snapshots/*/meta/...

사용 예:
    # Exp 1 대상 3개
    python script/build_instruction_meta.py \\
        --tasks CloseDrawer AdjustToasterOvenTemperature AdjustWaterTemperature

    # 전체 10개 (기본)
    python script/build_instruction_meta.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from extract_robocasa_images import DATASETS

HF_CACHE = Path.home() / ".cache/huggingface"
LEROBOT_ROOT = HF_CACHE / "lerobot"
HUB_ROOT = HF_CACHE / "hub"


def find_meta_dir(repo_id: str) -> Path:
    """LeRobot 캐시 우선, 없으면 hub snapshot 사용."""
    # repo_id = "kimz1121/<name>"
    lr = LEROBOT_ROOT / repo_id / "meta"
    if (lr / "tasks.parquet").exists():
        return lr

    safe = repo_id.replace("/", "--")
    snaps = sorted((HUB_ROOT / f"datasets--{safe}" / "snapshots").glob("*"))
    for s in snaps[::-1]:
        m = s / "meta"
        if (m / "tasks.parquet").exists():
            return m

    raise FileNotFoundError(
        f"meta/tasks.parquet not found for {repo_id} in either "
        f"{LEROBOT_ROOT} or {HUB_ROOT}"
    )


def build_for_task(task: str, repo_id: str, out_path: Path) -> dict:
    meta_dir = find_meta_dir(repo_id)
    tasks_df = pd.read_parquet(meta_dir / "tasks.parquet")
    eps_df = pd.read_parquet(
        meta_dir / "episodes" / "chunk-000" / "file-000.parquet"
    )

    # tasks.parquet: index 가 instruction 문자열, column 'task_index' 가 정수 ID
    # → {task_index: instruction} 으로 뒤집기
    instructions: dict[str, str] = {
        str(int(row.task_index)): str(idx)
        for idx, row in tasks_df.reset_index().set_index("task").iterrows()
    }
    # 위 코드가 헷갈리니 명시적으로 다시:
    instructions = {}
    for instr, ti in zip(tasks_df.index.tolist(), tasks_df["task_index"].tolist()):
        instructions[str(int(ti))] = str(instr)

    # episodes.parquet 'tasks' 컬럼은 list 형태 → 첫 원소가 그 episode 의 instruction
    ep_to_task_idx: dict[str, int] = {}
    instr_for_hist: list[str] = []
    for _, row in eps_df.iterrows():
        ep_idx = int(row["episode_index"])
        tlist = row["tasks"]
        # numpy ndarray or list
        first = tlist[0] if hasattr(tlist, "__len__") and len(tlist) > 0 else None
        if first is None:
            raise RuntimeError(f"episode {ep_idx} has empty tasks list")
        instr_for_hist.append(str(first))
        # task_index 역조회 (instruction → ti)
        matches = tasks_df.index[tasks_df.index == first].tolist()
        if not matches:
            raise RuntimeError(
                f"episode {ep_idx} instruction {first!r} not in tasks.parquet"
            )
        # tasks_df is indexed by instruction, get task_index value
        ti = int(tasks_df.loc[first, "task_index"])
        ep_to_task_idx[str(ep_idx)] = ti

    histogram = dict(Counter(instr_for_hist))

    payload = {
        "task": task,
        "dataset_repo": repo_id,
        "num_episodes": int(len(eps_df)),
        "num_unique_instructions": int(len(instructions)),
        "instructions": instructions,
        "episode_to_task_index": ep_to_task_idx,
        "histogram": histogram,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(DATASETS.keys()),
        default=sorted(DATASETS.keys()),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/patch_embeddings"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"output root: {args.output_root.resolve()}")
    print(f"tasks: {', '.join(args.tasks)}")
    print()

    for task in args.tasks:
        repo_id = DATASETS[task]
        out = args.output_root / f"{task}_instructions.json"
        try:
            payload = build_for_task(task, repo_id, out)
        except FileNotFoundError as e:
            print(f"  [skip] {task}: {e}")
            continue
        print(f"  ✓ {task}: {payload['num_episodes']} ep, "
              f"{payload['num_unique_instructions']} instruction(s) → {out}")
        for instr, cnt in payload["histogram"].items():
            short = instr if len(instr) <= 60 else instr[:57] + "..."
            print(f"      {cnt:>4} ep  | {short}")


if __name__ == "__main__":
    main()
