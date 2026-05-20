"""Cross-task sub-task contrastive (Exp 2, visual-only baseline).

10 task 의 patch_embeddings 를 모두 로드해서 **13-class fine-grained sub-task** 단위
contrastive 학습. trivial paraphrase (CloseCabinet door/doors, CloseFridge door/doors)
는 merged class 로 처리.

입력:
  - data/patch_embeddings/{Task}.hdf5         × 10
  - data/patch_embeddings/{Task}_instructions.json × 10
  - data/patch_embeddings/_unified_subtasks.json (mapping)

split:
  per-task × per-subtask stratified 80/20

사용:
    python script/contrastive_train_cross_task.py \\
        --unified-meta data/patch_embeddings/_unified_subtasks.json \\
        --out-dir runs/exp2_cross_task_visual

    # 빠른 sanity
    python script/contrastive_train_cross_task.py --epochs 5 --log-every 5
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

from contrastive_train import (
    BalancedBatchSampler,
    DisentangleModel,
    PatchClipDataset,
    SampleIndex,
    make_scheduler,
    run_epoch,
    run_probes,
)
from contrastive_train_subtask import load_instruction_meta


def build_cross_task_indices(
    h5_paths: list[Path],
    instr_dir: Path,
    unified_mapping: dict[str, int],
    train_ratio: float,
    cameras_filter: list[str] | None,
    rng: np.random.Generator,
) -> tuple[list[SampleIndex], list[SampleIndex], list[str], list[str]]:
    """모든 HDF5 + instruction JSON 로드, global sub-task id 기준 stratified split.

    Returns: train_idx, test_idx, cameras, task_order
    """
    cameras_all: list[str] | None = None
    train: list[SampleIndex] = []
    test: list[SampleIndex] = []
    task_order: list[str] = []

    for h5_path in h5_paths:
        task = h5_path.stem
        task_order.append(task)
        instr_path = instr_dir / f"{task}_instructions.json"
        if not instr_path.exists():
            raise SystemExit(f"missing instruction JSON: {instr_path}")
        instr_meta = load_instruction_meta(instr_path)
        ep_to_local_ti = {
            int(k): int(v) for k, v in instr_meta["episode_to_task_index"].items()
        }

        with h5py.File(h5_path, "r") as f:
            cams_in_file = [c for c in f.attrs["cameras"]]
            cams = cams_in_file if cameras_filter is None else [
                c for c in cams_in_file if c in set(cameras_filter)
            ]
            if cameras_all is None:
                cameras_all = cams
            elif cams != cameras_all:
                raise SystemExit(
                    f"camera 불일치: {h5_path} 의 {cams} vs 누적 {cameras_all}"
                )

            demos = sorted(f["data"].keys())

            # global sub-task id 별 episode 묶음
            per_subtask: dict[int, list[str]] = {}
            for demo_key in demos:
                ep_idx = int(demo_key.split("_")[1])
                local_ti = ep_to_local_ti.get(ep_idx)
                if local_ti is None:
                    raise SystemExit(
                        f"episode {ep_idx} from {h5_path} not in instruction JSON"
                    )
                global_id = unified_mapping[f"{task}/{local_ti}"]
                per_subtask.setdefault(global_id, []).append(demo_key)

            # stratified split (per sub-task, within this task)
            train_demos: set[str] = set()
            for gid, dlist in per_subtask.items():
                arr = np.array(dlist)
                rng.shuffle(arr)
                n_train = int(math.floor(train_ratio * len(arr)))
                for d in arr[:n_train]:
                    train_demos.add(str(d))

            for demo_key in demos:
                ep_idx = int(demo_key.split("_")[1])
                local_ti = ep_to_local_ti[ep_idx]
                gid = unified_mapping[f"{task}/{local_ti}"]
                clip_keys = sorted(f[f"data/{demo_key}"].keys())
                target = train if demo_key in train_demos else test
                for clip_key in clip_keys:
                    clip_id = int(clip_key.split("_")[1])
                    for cam_id, cam in enumerate(cams):
                        target.append(
                            SampleIndex(
                                h5_path=str(h5_path),
                                task_id=gid,              # global sub-task id
                                camera=cam,
                                camera_id=cam_id,
                                ep_idx=ep_idx,
                                demo_key=demo_key,
                                clip_id=clip_id,
                                clip_key=clip_key,
                            )
                        )

    if cameras_all is None:
        raise SystemExit("HDF5 가 비었거나 camera 정보 없음")

    # per-subtask 분포 출력
    def count(idx_list):
        c: dict[int, int] = {}
        for s in idx_list:
            c[s.task_id] = c.get(s.task_id, 0) + 1
        return c

    print(f"[INFO] cameras={cameras_all}")
    print(f"[INFO] task_order={task_order}")
    print(f"[INFO] train clips per sub-task: {count(train)}")
    print(f"[INFO] test  clips per sub-task: {count(test)}")
    print(f"[INFO] train={len(train)} test={len(test)}")
    return train, test, cameras_all, task_order


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--patch-h5-root", type=Path, default=Path("data/patch_embeddings"))
    p.add_argument(
        "--unified-meta",
        type=Path,
        default=Path("data/patch_embeddings/_unified_subtasks.json"),
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="기본: unified-meta 의 mapping 에 있는 모든 task",
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
        help="기본: runs/exp2_cross_task_visual_{timestamp}",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--probe-every", type=int, default=1)
    p.add_argument("--probe-k", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("runs") / f"exp2_cross_task_visual_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] out_dir={out_dir}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- unified mapping ----
    if not args.unified_meta.exists():
        raise SystemExit(
            f"unified meta 없음: {args.unified_meta}. "
            "먼저 build_unified_subtask_meta.py 실행"
        )
    unified = json.loads(args.unified_meta.read_text())
    unified_mapping: dict[str, int] = unified["mapping"]
    num_subtasks: int = int(unified["num_subtasks"])
    subtask_id_to_label = {int(k): v for k, v in unified["subtask_id_to_label"].items()}
    print(f"[INFO] unified subtasks: {num_subtasks}")
    for sid in sorted(subtask_id_to_label.keys()):
        print(f"         {sid:>2}: {subtask_id_to_label[sid][:80]}")

    # ---- HDF5 모으기 ----
    tasks_from_mapping = sorted({k.split("/")[0] for k in unified_mapping.keys()})
    tasks = args.tasks or tasks_from_mapping
    h5_paths = []
    for t in tasks:
        p = args.patch_h5_root / f"{t}.hdf5"
        if not p.exists():
            raise SystemExit(f"HDF5 없음: {p}")
        h5_paths.append(p)
    print(f"[INFO] using {len(h5_paths)} HDF5 files")

    # ---- 데이터 ----
    split_rng = np.random.default_rng(args.seed)
    train_idx, test_idx, cameras, task_order = build_cross_task_indices(
        h5_paths, args.patch_h5_root, unified_mapping,
        args.train_ratio, args.cameras, split_rng,
    )
    num_cameras = len(cameras)

    train_ds = PatchClipDataset(train_idx)
    test_ds = PatchClipDataset(test_idx)

    sampler_rng = np.random.default_rng(args.seed)
    train_sampler = BalancedBatchSampler(
        train_idx, args.batch_size, num_subtasks, num_cameras, sampler_rng
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    probe_train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )

    # ---- Model ----
    with h5py.File(h5_paths[0], "r") as f:
        d_model = int(f.attrs["dim"])
    model = DisentangleModel(
        d_model=d_model, num_queries=args.num_queries,
        d_task=args.d_task, d_nuis=args.d_nuis,
        hidden=args.hidden, num_heads=args.num_heads, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] model params={n_params:,}  device={device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs)

    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config = {
        "experiment": "exp2_cross_task_visual_only",
        "args": args_dict,
        "tasks": task_order,
        "num_subtasks": num_subtasks,
        "subtask_id_to_label": {int(k): v for k, v in subtask_id_to_label.items()},
        "unified_mapping": unified_mapping,
        "cameras": cameras,
        "h5_paths": [str(p) for p in h5_paths],
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
            model, train_loader, optimizer, device,
            temperature=args.temperature,
            lambda_nuis=args.lambda_nuis, lambda_ortho=args.lambda_ortho,
            log_every=args.log_every, epoch=epoch, log_fp=log_fp,
        )
        test_stats = run_epoch(
            model, test_loader, None, device,
            temperature=args.temperature,
            lambda_nuis=args.lambda_nuis, lambda_ortho=args.lambda_ortho,
        )
        scheduler.step()

        probe_stats: dict[str, float] | None = None
        if args.probe_every and (epoch + 1) % args.probe_every == 0:
            probe_stats = run_probes(
                model, probe_train_loader, test_loader, device, k=args.probe_k
            )

        row = {"epoch": epoch, "phase": "epoch_summary",
               "train": train_stats, "test": test_stats, "probe": probe_stats,
               "lr": optimizer.param_groups[0]["lr"]}
        log_fp.write(json.dumps(row) + "\n")
        log_fp.flush()

        print(
            f"  [train] loss={train_stats['loss']:.4f}  "
            f"task={train_stats['L_task']:.4f}  nuis={train_stats['L_nuis']:.4f}  "
            f"orth={train_stats['L_orth']:.4f}"
        )
        print(
            f"  [test]  loss={test_stats['loss']:.4f}  "
            f"task={test_stats['L_task']:.4f}  nuis={test_stats['L_nuis']:.4f}"
        )
        if probe_stats is not None:
            print(
                f"  [probe] subtask@z_t={probe_stats['task_on_z_task']:.3f}  "
                f"cam@z_t={probe_stats['cam_on_z_task']:.3f}  |  "
                f"cam@z_n={probe_stats['cam_on_z_nuis']:.3f}  "
                f"subtask@z_n={probe_stats['task_on_z_nuis']:.3f}"
            )

        if test_stats["loss"] < best_test_loss:
            best_test_loss = test_stats["loss"]
            torch.save(
                {"model": model.state_dict(),
                 "epoch": epoch, "test_stats": test_stats, "config": config},
                out_dir / "best.pt",
            )
        if (epoch + 1) % args.ckpt_every == 0 or epoch + 1 == args.epochs:
            torch.save(
                {"model": model.state_dict(),
                 "epoch": epoch, "test_stats": test_stats, "config": config},
                out_dir / f"ckpt_epoch_{epoch:02d}.pt",
            )

    log_fp.close()
    print(f"\n[done] best test loss = {best_test_loss:.4f}  out_dir={out_dir}")


if __name__ == "__main__":
    main()
