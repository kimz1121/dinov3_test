"""10 task 의 instruction JSON 을 통합해서 13-class sub-task mapping 생성.

Trivial paraphrase (단/복수 차이 정도) 는 한 class 로 merge:
  - CloseCabinet: door / doors   → 1 class
  - CloseFridge:  door / doors   → 1 class
나머지는 각 instruction 마다 별도 class.

기대 결과 (10 task, 13 sub-task):
  Adjust  ↑↓ 2종, Water hot↔cold 2종, Drawer L/R 2종 = 6
  + 단일 instruction 5종 (CheesyBread, BlenderLid, ElectricKettleLid, Dishwasher, FridgeDrawer) = 5
  + merged 2종 (Cabinet, Fridge) = 2
  ──────────────────────────────────────────────────────
                                              합계  13

출력:
  data/patch_embeddings/_unified_subtasks.json
  {
    "mapping": {
      "AdjustToasterOvenTemperature/0": 0,
      "AdjustToasterOvenTemperature/1": 1,
      "AdjustWaterTemperature/0": 2,
      ...
      "CloseCabinet/0": 6,    # merged
      "CloseCabinet/1": 6,    # merged
      ...
    },
    "subtask_id_to_label": {
      "0": "AdjustToasterOvenTemperature : Increase the toaster oven temperature.",
      ...
      "6": "CloseCabinet : (merged door/doors)",
      ...
    },
    "subtask_id_to_episode_count": {"0": 46, "1": 61, ...},
    "trivial_merged_tasks": ["CloseCabinet", "CloseFridge"]
  }

사용:
  python script/build_unified_subtask_meta.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# 사용자가 노션에서 분류한 대로
TRIVIAL_PARAPHRASE_TASKS = {"CloseCabinet", "CloseFridge"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--instr-dir",
        type=Path,
        default=Path("data/patch_embeddings"),
        help="*_instructions.json 위치",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/patch_embeddings/_unified_subtasks.json"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    json_files = sorted(args.instr_dir.glob("*_instructions.json"))
    print(f"[INFO] found {len(json_files)} instruction JSON files")

    mapping: dict[str, int] = {}
    subtask_id_to_label: dict[int, str] = {}
    subtask_id_to_episode_count: dict[int, int] = {}
    next_id = 0

    # task 이름 정렬해서 순서 안정화
    tasks_data = []
    for f in json_files:
        d = json.loads(f.read_text())
        tasks_data.append(d)
    tasks_data.sort(key=lambda d: d["task"])

    for d in tasks_data:
        task = d["task"]
        instrs = d["instructions"]
        hist = d["histogram"]
        sorted_keys = sorted(instrs.keys(), key=int)

        if task in TRIVIAL_PARAPHRASE_TASKS:
            # 모든 instruction 을 한 class 로
            gid = next_id
            next_id += 1
            total_ep = sum(hist.values())
            label_text = f"{task} : (merged {len(instrs)} paraphrase)"
            subtask_id_to_label[gid] = label_text
            subtask_id_to_episode_count[gid] = total_ep
            for k in sorted_keys:
                mapping[f"{task}/{k}"] = gid
            print(f"  [merge] {task}: {len(instrs)} instr → 1 sub-task (id={gid}, {total_ep} ep)")
        else:
            for k in sorted_keys:
                gid = next_id
                next_id += 1
                instr_text = instrs[k]
                short = instr_text if len(instr_text) <= 60 else instr_text[:57] + "..."
                subtask_id_to_label[gid] = f"{task} : {instr_text}"
                subtask_id_to_episode_count[gid] = hist.get(instr_text, 0)
                mapping[f"{task}/{k}"] = gid
                print(f"  [split] {task}/{k} (id={gid}, {hist.get(instr_text, 0)} ep): {short}")

    print(f"\n[INFO] total sub-tasks: {next_id}")

    payload = {
        "mapping": mapping,
        "subtask_id_to_label": {str(k): v for k, v in subtask_id_to_label.items()},
        "subtask_id_to_episode_count": {str(k): v for k, v in subtask_id_to_episode_count.items()},
        "num_subtasks": next_id,
        "trivial_merged_tasks": sorted(TRIVIAL_PARAPHRASE_TASKS),
        "source_files": [str(f) for f in json_files],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n[done] → {args.output}")


if __name__ == "__main__":
    main()
