"""Within-task sub-task contrastive (Exp 1a, visual-only baseline).

기존 contrastive_train.py 와 같은 모델/loss/scheduler 를 쓰되, label 을
**task 이름 대신 episode 의 instruction (sub-task)** 으로 바꿔서 1 개 task 안의
2 개 instruction 을 분리하는 학습을 수행한다.

입력:
  - data/patch_embeddings/{Task}.hdf5
  - data/patch_embeddings/{Task}_instructions.json  (build_instruction_meta.py 산출)

Split:
  episode 별 sub-task stratified split. 즉 sub-task 0/1 각각 train 80% / test 20%.

사용 예:
    # CloseDrawer L/R minimal pair, 기본 하이퍼파라미터
    python script/contrastive_train_subtask.py --task CloseDrawer

    # epochs 작게, 빠른 sanity check
    python script/contrastive_train_subtask.py --task CloseDrawer --epochs 5 --log-every 5

    # 다른 pair
    python script/contrastive_train_subtask.py --task AdjustToasterOvenTemperature
    python script/contrastive_train_subtask.py --task AdjustWaterTemperature
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

# 기존 train 스크립트의 빌딩블록 재사용
from contrastive_train import (
    SampleIndex,
    PatchClipDataset,
    BalancedBatchSampler,
    DisentangleModel,
    run_epoch,
    run_probes,
    make_scheduler,
)


# ---------------------------------------------------------------------------
# Sub-task index builder
# ---------------------------------------------------------------------------


def load_instruction_meta(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


def build_subtask_indices(
    h5_path: Path,
    instr_meta: dict,
    train_ratio: float,
    cameras_filter: list[str] | None,
    rng: np.random.Generator,
) -> tuple[list[SampleIndex], list[SampleIndex], dict[int, str], list[str]]:
    """1 개 task HDF5 + instruction JSON → sub-task stratified train/test split.

    SampleIndex.task_id 자리에 **sub_task_id** (instruction task_index) 를 넣는다.
    Returns:
        train_idx, test_idx, subtask_id_to_instruction, cameras
    """
    ep_to_subtask: dict[int, int] = {
        int(k): int(v) for k, v in instr_meta["episode_to_task_index"].items()
    }
    subtask_to_instr: dict[int, str] = {
        int(k): str(v) for k, v in instr_meta["instructions"].items()
    }

    with h5py.File(h5_path, "r") as f:
        cams_in_file = [c for c in f.attrs["cameras"]]
        cams = cams_in_file if cameras_filter is None else [
            c for c in cams_in_file if c in set(cameras_filter)
        ]
        demos = sorted(f["data"].keys())

        # sub-task 별 episode 묶음 (정렬된 demo 순서 유지)
        per_subtask: dict[int, list[str]] = {}
        for demo_key in demos:
            ep_idx = int(demo_key.split("_")[1])
            if ep_idx not in ep_to_subtask:
                raise SystemExit(
                    f"episode {ep_idx} from {h5_path} not in instruction JSON; "
                    f"check {instr_meta.get('task')!r} instruction file"
                )
            per_subtask.setdefault(ep_to_subtask[ep_idx], []).append(demo_key)

        # stratified split: 각 sub-task 별로 episode shuffle 후 80/20
        train_demos: set[str] = set()
        for stid, dlist in per_subtask.items():
            arr = np.array(dlist)
            rng.shuffle(arr)
            n_train = int(math.floor(train_ratio * len(arr)))
            for d in arr[:n_train]:
                train_demos.add(str(d))

        train: list[SampleIndex] = []
        test: list[SampleIndex] = []
        for demo_key in demos:
            ep_idx = int(demo_key.split("_")[1])
            sub_id = ep_to_subtask[ep_idx]
            clip_keys = sorted(f[f"data/{demo_key}"].keys())
            target = train if demo_key in train_demos else test
            for clip_key in clip_keys:
                clip_id = int(clip_key.split("_")[1])
                for cam_id, cam in enumerate(cams):
                    target.append(
                        SampleIndex(
                            h5_path=str(h5_path),
                            task_id=sub_id,        # ← sub-task id 자리
                            camera=cam,
                            camera_id=cam_id,
                            ep_idx=ep_idx,
                            demo_key=demo_key,
                            clip_id=clip_id,
                            clip_key=clip_key,
                        )
                    )

    # 통계 출력
    def count_by_subtask(idx_list):
        c: dict[int, int] = {}
        for s in idx_list:
            c[s.task_id] = c.get(s.task_id, 0) + 1
        return c

    print(f"[INFO] cameras={cams}")
    print(f"[INFO] sub_task → instruction:")
    for sid, instr in subtask_to_instr.items():
        short = instr if len(instr) <= 70 else instr[:67] + "..."
        print(f"         {sid}: {short}")
    print(f"[INFO] train clips per sub_task: {count_by_subtask(train)}")
    print(f"[INFO] test  clips per sub_task: {count_by_subtask(test)}")
    print(f"[INFO] train={len(train)} test={len(test)}")
    return train, test, subtask_to_instr, cams


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--task",
        required=True,
        help="단일 task 이름 (예: CloseDrawer). HDF5 와 instruction JSON 의 stem.",
    )
    p.add_argument(
        "--patch-h5-root",
        type=Path,
        default=Path("data/patch_embeddings"),
    )
    p.add_argument(
        "--instructions-json",
        type=Path,
        default=None,
        help="기본: {patch_h5_root}/{task}_instructions.json",
    )
    p.add_argument("--cameras", nargs="+", default=None)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--num-queries", type=int, default=4)
    p.add_argument("--d-task", type=int, default=128)
    p.add_argument("--d-nuis", type=int, default=64)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lambda-nuis", type=float, default=1.0)
    p.add_argument("--lambda-ortho", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="기본: runs/exp1_{task}_visual_{timestamp}",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--probe-every", type=int, default=1)
    p.add_argument("--probe-k", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("runs") / f"exp1_{args.task}_visual_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] out_dir={out_dir}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 입력 경로 ----
    h5_path = args.patch_h5_root / f"{args.task}.hdf5"
    if not h5_path.exists():
        raise SystemExit(f"HDF5 없음: {h5_path}")
    instr_json = args.instructions_json or (
        args.patch_h5_root / f"{args.task}_instructions.json"
    )
    if not instr_json.exists():
        raise SystemExit(
            f"instruction JSON 없음: {instr_json} (build_instruction_meta.py 먼저 실행)"
        )

    print(f"[INFO] task={args.task}  h5={h5_path}  instr={instr_json}")
    instr_meta = load_instruction_meta(instr_json)
    print(
        f"[INFO] {instr_meta['num_episodes']} episodes, "
        f"{instr_meta['num_unique_instructions']} sub-tasks"
    )

    # ---- 데이터 ----
    split_rng = np.random.default_rng(args.seed)
    train_idx, test_idx, subtask_to_instr, cameras = build_subtask_indices(
        h5_path, instr_meta, args.train_ratio, args.cameras, split_rng
    )
    num_subtasks = len(subtask_to_instr)
    num_cameras = len(cameras)

    train_ds = PatchClipDataset(train_idx)
    test_ds = PatchClipDataset(test_idx)

    sampler_rng = np.random.default_rng(args.seed)
    train_sampler = BalancedBatchSampler(
        train_idx, args.batch_size, num_subtasks, num_cameras, sampler_rng
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    probe_train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # ---- 모델 ----
    with h5py.File(h5_path, "r") as f:
        d_model = int(f.attrs["dim"])
    model = DisentangleModel(
        d_model=d_model,
        num_queries=args.num_queries,
        d_task=args.d_task,
        d_nuis=args.d_nuis,
        hidden=args.hidden,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] model params={n_params:,}  device={device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs)

    # ---- config 저장 ----
    args_dict = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
    }
    config = {
        "experiment": "exp1_within_task_visual_only",
        "args": args_dict,
        "task": args.task,
        "subtask_to_instruction": {int(k): v for k, v in subtask_to_instr.items()},
        "cameras": cameras,
        "h5_path": str(h5_path),
        "instructions_json": str(instr_json),
        "d_model": d_model,
        "model_config": model.config,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    log_fp = open(out_dir / "log.jsonl", "w")

    # ---- 학습 ----
    best_test_loss = float("inf")
    for epoch in range(args.epochs):
        print(
            f"\n=== epoch {epoch:02d}/{args.epochs}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e} ==="
        )
        train_stats = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            temperature=args.temperature,
            lambda_nuis=args.lambda_nuis,
            lambda_ortho=args.lambda_ortho,
            log_every=args.log_every,
            epoch=epoch,
            log_fp=log_fp,
        )
        test_stats = run_epoch(
            model,
            test_loader,
            None,
            device,
            temperature=args.temperature,
            lambda_nuis=args.lambda_nuis,
            lambda_ortho=args.lambda_ortho,
        )
        scheduler.step()

        probe_stats: dict[str, float] | None = None
        if args.probe_every and (epoch + 1) % args.probe_every == 0:
            probe_stats = run_probes(
                model, probe_train_loader, test_loader, device, k=args.probe_k
            )

        row = {
            "epoch": epoch,
            "phase": "epoch_summary",
            "train": train_stats,
            "test": test_stats,
            "probe": probe_stats,
            "lr": optimizer.param_groups[0]["lr"],
        }
        log_fp.write(json.dumps(row) + "\n")
        log_fp.flush()

        print(
            f"  [train] loss={train_stats['loss']:.4f}  "
            f"task={train_stats['L_task']:.4f}  "
            f"nuis={train_stats['L_nuis']:.4f}  "
            f"orth={train_stats['L_orth']:.4f}"
        )
        print(
            f"  [test]  loss={test_stats['loss']:.4f}  "
            f"task={test_stats['L_task']:.4f}  "
            f"nuis={test_stats['L_nuis']:.4f}  "
            f"orth={test_stats['L_orth']:.4f}"
        )
        if probe_stats is not None:
            print(
                f"  [probe] subtask@z_t={probe_stats['task_on_z_task']:.3f}  "
                f"cam@z_t={probe_stats['cam_on_z_task']:.3f}  "
                f"(↓ disentangle)  |  "
                f"cam@z_n={probe_stats['cam_on_z_nuis']:.3f}  "
                f"subtask@z_n={probe_stats['task_on_z_nuis']:.3f}  (↓ disentangle)"
            )

        if test_stats["loss"] < best_test_loss:
            best_test_loss = test_stats["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "test_stats": test_stats,
                    "config": config,
                },
                out_dir / "best.pt",
            )

        if (epoch + 1) % args.ckpt_every == 0 or epoch + 1 == args.epochs:
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "test_stats": test_stats,
                    "config": config,
                },
                out_dir / f"ckpt_epoch_{epoch:02d}.pt",
            )

    log_fp.close()
    print(f"\n[done] best test loss = {best_test_loss:.4f}  out_dir={out_dir}")


if __name__ == "__main__":
    main()
