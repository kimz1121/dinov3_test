"""Cross-task sub-task contrastive WITH vision-language alignment (Exp 2b).

contrastive_train_cross_task.py + contrastive_train_subtask_vl.py 합본:
  - 10 task × 13 sub-task 한꺼번에 학습
  - 각 sub-task 의 instruction text 를 CLIP encode → text proto (13, 512)
  - text projection 으로 (13, d_task) → z_task 와 alignment

trivial-merged sub-task (CloseCabinet, CloseFridge) 의 text 는 첫 번째
instruction 으로 대표 (둘 다 같은 의미라 가정).

사용:
    python script/contrastive_train_cross_task_vl.py --vl-loss alignment_infonce
    python script/contrastive_train_cross_task_vl.py --vl-loss anchor_regression --freeze-text-proj
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from contrastive_train import (
    BalancedBatchSampler,
    DisentangleModel,
    PatchClipDataset,
    compute_embeddings,
    knn_accuracy,
    make_scheduler,
    ortho_loss,
    supcon_loss,
)
from contrastive_train_cross_task import build_cross_task_indices
from contrastive_train_subtask import load_instruction_meta
from contrastive_train_subtask_vl import (
    VLModel,
    alignment_infonce_loss,
    anchor_regression_loss,
    encode_texts_with_clip,
)


def build_subtask_instruction_texts(
    unified_mapping: dict[str, int],
    instr_dir: Path,
) -> dict[int, str]:
    """global sub-task id → 대표 instruction 문자열.

    merged sub-task 는 첫 번째 instruction (정렬 기준) 사용.
    """
    # subtask id → 후보 (task, local_ti) 모음
    sid_candidates: dict[int, list[tuple[str, int]]] = {}
    for key, sid in unified_mapping.items():
        task, local_ti_str = key.split("/")
        sid_candidates.setdefault(sid, []).append((task, int(local_ti_str)))

    sid_to_text: dict[int, str] = {}
    for sid, cands in sid_candidates.items():
        # 첫 후보 (task 이름 알파벳 + local_ti 작은 순) 사용
        cands.sort()
        task, local_ti = cands[0]
        instr_meta = load_instruction_meta(instr_dir / f"{task}_instructions.json")
        sid_to_text[sid] = instr_meta["instructions"][str(local_ti)]
    return sid_to_text


def run_epoch_vl(
    vl_model: VLModel,
    text_protos: torch.Tensor,
    loader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    *,
    temperature: float,
    lambda_nuis: float,
    lambda_ortho: float,
    lambda_vl: float,
    vl_loss_kind: str,
    log_every: int = 0,
    epoch: int = 0,
    log_fp=None,
) -> dict[str, float]:
    is_train = optimizer is not None
    vl_model.train(is_train)
    sums = {"loss": 0.0, "L_task": 0.0, "L_nuis": 0.0, "L_orth": 0.0, "L_vl": 0.0}
    n_steps = 0

    text_protos_dev = text_protos.to(device)
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for step, batch in enumerate(loader):
            patches, task_id, cam_id, _, _ = batch
            patches = patches.to(device, non_blocking=True)
            task_id = task_id.to(device)
            cam_id = cam_id.to(device)

            z_task, z_nuis = vl_model(patches)
            z_text = vl_model.project_text(text_protos_dev)

            L_task = supcon_loss(z_task, task_id, temperature)
            L_nuis = supcon_loss(z_nuis, cam_id, temperature)
            L_orth = ortho_loss(z_task, z_nuis)
            if vl_loss_kind == "alignment_infonce":
                L_vl = alignment_infonce_loss(z_task, task_id, z_text, temperature)
            elif vl_loss_kind == "anchor_regression":
                L_vl = anchor_regression_loss(z_task, task_id, z_text)
            else:
                raise ValueError(f"unknown vl_loss_kind: {vl_loss_kind}")
            loss = L_task + lambda_nuis * L_nuis + lambda_ortho * L_orth + lambda_vl * L_vl

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            for k, v in zip(
                ["loss", "L_task", "L_nuis", "L_orth", "L_vl"],
                [loss, L_task, L_nuis, L_orth, L_vl],
            ):
                sums[k] += float(v.item())
            n_steps += 1

            if is_train and log_every and step % log_every == 0:
                print(
                    f"  ep{epoch:02d} step{step:03d}  "
                    f"loss={loss.item():.4f}  task={L_task.item():.4f}  "
                    f"nuis={L_nuis.item():.4f}  vl={L_vl.item():.4f}"
                )
                if log_fp is not None:
                    log_fp.write(json.dumps({
                        "epoch": epoch, "step": step, "phase": "train",
                        "loss": loss.item(), "L_task": L_task.item(),
                        "L_nuis": L_nuis.item(), "L_orth": L_orth.item(),
                        "L_vl": L_vl.item(),
                    }) + "\n")
                    log_fp.flush()
    return {k: v / max(n_steps, 1) for k, v in sums.items()}


@torch.no_grad()
def run_probes_vl(vl_model, text_protos, train_loader, test_loader, device, k=10):
    zt_tr, zn_tr, yt_tr, yc_tr = compute_embeddings(vl_model.visual, train_loader, device)
    zt_te, zn_te, yt_te, yc_te = compute_embeddings(vl_model.visual, test_loader, device)
    z_text = vl_model.project_text(text_protos.to(device)).cpu()
    logits = zt_te @ z_text.T
    pred_text = logits.argmax(dim=-1)
    text_align_acc = (pred_text == yt_te).float().mean().item()
    return {
        "task_on_z_task": knn_accuracy(zt_tr, yt_tr, zt_te, yt_te, k),
        "cam_on_z_task":  knn_accuracy(zt_tr, yc_tr, zt_te, yc_te, k),
        "task_on_z_nuis": knn_accuracy(zn_tr, yt_tr, zn_te, yt_te, k),
        "cam_on_z_nuis":  knn_accuracy(zn_tr, yc_tr, zn_te, yc_te, k),
        "text_align_acc": text_align_acc,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patch-h5-root", type=Path, default=Path("data/patch_embeddings"))
    p.add_argument("--unified-meta", type=Path,
                   default=Path("data/patch_embeddings/_unified_subtasks.json"))
    p.add_argument("--vl-loss", choices=["alignment_infonce", "anchor_regression"],
                   required=True)
    p.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
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
    p.add_argument("--lambda-vl", type=float, default=1.0)
    p.add_argument("--text-proj-hidden", type=int, default=256)
    p.add_argument("--freeze-text-proj", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--probe-every", type=int, default=1)
    p.add_argument("--probe-k", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"vl_{args.vl_loss}"
    if args.freeze_text_proj:
        suffix += "_frozen"
    out_dir = args.out_dir or Path("runs") / f"exp2_cross_task_{suffix}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] out_dir={out_dir}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    unified = json.loads(args.unified_meta.read_text())
    unified_mapping: dict[str, int] = unified["mapping"]
    num_subtasks: int = int(unified["num_subtasks"])
    subtask_id_to_label = {int(k): v for k, v in unified["subtask_id_to_label"].items()}

    # ---- 각 sub-task 의 대표 instruction → CLIP encode ----
    sid_to_text = build_subtask_instruction_texts(unified_mapping, args.patch_h5_root)
    instruction_strs = [sid_to_text[i] for i in range(num_subtasks)]
    print(f"[INFO] encoding {num_subtasks} sub-task texts with CLIP {args.clip_model}")
    for i, s in enumerate(instruction_strs):
        short = s if len(s) <= 70 else s[:67] + "..."
        print(f"         [{i:>2}] {short}")
    text_protos, clip_info = encode_texts_with_clip(
        instruction_strs, model_id=args.clip_model, device=device
    )
    print(f"[INFO] text_protos shape={tuple(text_protos.shape)}")

    tasks_from_mapping = sorted({k.split("/")[0] for k in unified_mapping.keys()})
    h5_paths = [args.patch_h5_root / f"{t}.hdf5" for t in tasks_from_mapping]
    missing = [p for p in h5_paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing HDF5: {missing}")
    print(f"[INFO] using {len(h5_paths)} HDF5 files")

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
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device == "cuda"))
    probe_train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=(device == "cuda"))

    with h5py.File(h5_paths[0], "r") as f:
        d_model = int(f.attrs["dim"])
    visual_model = DisentangleModel(
        d_model=d_model, num_queries=args.num_queries,
        d_task=args.d_task, d_nuis=args.d_nuis,
        hidden=args.hidden, num_heads=args.num_heads, dropout=args.dropout,
    )
    vl_model = VLModel(visual_model, text_dim=clip_info["dim"],
                        d_task=args.d_task, hidden=args.text_proj_hidden).to(device)

    if args.freeze_text_proj:
        for p in vl_model.text_proj_net.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            z_text_check = vl_model.project_text(text_protos.to(device)).cpu()
            # pair cos mean (off-diag)
            n_st = z_text_check.shape[0]
            G = z_text_check @ z_text_check.T
            pair_cos = float(G[~np.eye(n_st, dtype=bool)].mean())
            print(f"[INFO] freeze-text-proj enabled. z_text pair cos (mean off-diag) = {pair_cos:.4f}")

    n_train = sum(p.numel() for p in vl_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in vl_model.parameters())
    print(f"[INFO] trainable params={n_train:,} / total {n_total:,}  device={device}")

    optimizer = torch.optim.AdamW(
        [p for p in vl_model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs)

    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config = {
        "experiment": "exp2_cross_task_visual_language",
        "args": args_dict,
        "vl_loss": args.vl_loss,
        "clip_model": args.clip_model,
        "tasks": task_order,
        "num_subtasks": num_subtasks,
        "subtask_id_to_label": {int(k): v for k, v in subtask_id_to_label.items()},
        "subtask_id_to_text": sid_to_text,
        "unified_mapping": unified_mapping,
        "cameras": cameras,
        "h5_paths": [str(p) for p in h5_paths],
        "d_model": d_model,
        "model_config": visual_model.config,
        "text_dim": clip_info["dim"],
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    torch.save(text_protos, out_dir / "text_protos.pt")
    log_fp = open(out_dir / "log.jsonl", "w")

    best_test_loss = float("inf")
    for epoch in range(args.epochs):
        print(f"\n=== epoch {epoch:02d}/{args.epochs}  lr={optimizer.param_groups[0]['lr']:.2e} ===")
        train_stats = run_epoch_vl(
            vl_model, text_protos, train_loader, optimizer, device,
            temperature=args.temperature,
            lambda_nuis=args.lambda_nuis, lambda_ortho=args.lambda_ortho,
            lambda_vl=args.lambda_vl, vl_loss_kind=args.vl_loss,
            log_every=args.log_every, epoch=epoch, log_fp=log_fp,
        )
        test_stats = run_epoch_vl(
            vl_model, text_protos, test_loader, None, device,
            temperature=args.temperature,
            lambda_nuis=args.lambda_nuis, lambda_ortho=args.lambda_ortho,
            lambda_vl=args.lambda_vl, vl_loss_kind=args.vl_loss,
        )
        scheduler.step()

        probe_stats = None
        if args.probe_every and (epoch + 1) % args.probe_every == 0:
            probe_stats = run_probes_vl(vl_model, text_protos,
                                         probe_train_loader, test_loader, device, k=args.probe_k)
        log_fp.write(json.dumps({
            "epoch": epoch, "phase": "epoch_summary",
            "train": train_stats, "test": test_stats, "probe": probe_stats,
            "lr": optimizer.param_groups[0]["lr"],
        }) + "\n")
        log_fp.flush()

        print(f"  [train] loss={train_stats['loss']:.4f}  task={train_stats['L_task']:.4f}  "
              f"nuis={train_stats['L_nuis']:.4f}  vl={train_stats['L_vl']:.4f}")
        print(f"  [test]  loss={test_stats['loss']:.4f}  task={test_stats['L_task']:.4f}  "
              f"vl={test_stats['L_vl']:.4f}")
        if probe_stats:
            print(f"  [probe] subtask@z_t={probe_stats['task_on_z_task']:.3f}  "
                  f"text_align={probe_stats['text_align_acc']:.3f}  "
                  f"cam@z_t={probe_stats['cam_on_z_task']:.3f}  |  "
                  f"cam@z_n={probe_stats['cam_on_z_nuis']:.3f}")

        if test_stats["loss"] < best_test_loss:
            best_test_loss = test_stats["loss"]
            torch.save({"vl_model": vl_model.state_dict(),
                        "epoch": epoch, "test_stats": test_stats, "config": config},
                       out_dir / "best.pt")
        if (epoch + 1) % args.ckpt_every == 0 or epoch + 1 == args.epochs:
            torch.save({"vl_model": vl_model.state_dict(),
                        "epoch": epoch, "test_stats": test_stats, "config": config},
                       out_dir / f"ckpt_epoch_{epoch:02d}.pt")

    log_fp.close()
    print(f"\n[done] best test loss = {best_test_loss:.4f}  out_dir={out_dir}")


if __name__ == "__main__":
    main()
