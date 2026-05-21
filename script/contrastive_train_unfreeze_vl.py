"""P1 + B-1 — DINOv3 last-N unfreeze + VL alignment_infonce.

P1 (visual unfreeze) 로 knn 0.767 → 0.826 까지 끌어올렸지만 motion pair
(toaster inc/dec, water hot/cold) 의 recall 은 약한 개선에 그침. 시나리오 B-1:
이제 visual feature 가 학습 가능하므로 text alignment 의 gradient 가 motion
pair 의 representation 까지 흘려보낼 수 있는지 검증.

기존 Exp 2 VL alignment_infonce 와 차이:
  - frozen patch HDF5 대신 raw image + DINOv3 forward
  - DINOv3 last N block trainable
  - 나머지는 동일 (text_proj_net + alignment_infonce + SupCon + ortho)

학습 후 평가:
  - knn(10) overall + hot pair recall
  - vs P1 visual-only (0.826) — 추가 개선 여부가 핵심
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from contrastive_train import (
    BalancedBatchSampler,
    SampleIndex,
    knn_accuracy,
    make_scheduler,
    ortho_loss,
    supcon_loss,
)
from contrastive_train_cross_task import build_cross_task_indices
from contrastive_train_subtask_vl import (
    alignment_infonce_loss,
    encode_texts_with_clip,
)
from contrastive_train_unfreeze import RawClipDataset, UnfreezeDinoModel


class UnfreezeVLModel(nn.Module):
    """UnfreezeDinoModel + text_proj_net (alignment_infonce 용)."""

    def __init__(
        self,
        text_dim: int,
        text_proj_hidden: int = 256,
        **dino_kwargs,
    ):
        super().__init__()
        self.visual = UnfreezeDinoModel(**dino_kwargs)
        d_task = self.visual.config["d_task"]
        self.text_proj_net = nn.Sequential(
            nn.Linear(text_dim, text_proj_hidden),
            nn.GELU(),
            nn.Linear(text_proj_hidden, d_task),
            nn.LayerNorm(d_task),
        )
        self.config = dict(self.visual.config)
        self.config.update(
            text_dim=text_dim,
            text_proj_hidden=text_proj_hidden,
        )

    def forward(self, images):
        return self.visual(images)

    def project_text(self, text_emb):
        z = self.text_proj_net(text_emb)
        return F.normalize(z, dim=-1)


def run_epoch_unfreeze_vl(
    model: UnfreezeVLModel,
    text_protos: torch.Tensor,
    loader,
    optimizer,
    device: str,
    *,
    temperature: float,
    lambda_nuis: float,
    lambda_ortho: float,
    lambda_vl: float,
    autocast_dtype,
    accum_steps: int = 1,
    log_every: int = 0,
    epoch: int = 0,
    log_fp=None,
):
    is_train = optimizer is not None
    model.train(is_train)
    sums = {"loss": 0.0, "L_task": 0.0, "L_nuis": 0.0, "L_orth": 0.0, "L_vl": 0.0}
    n_steps = 0
    text_protos_dev = text_protos.to(device)
    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            imgs, task_id, cam_id, _, _ = batch
            imgs = imgs.to(device, non_blocking=True)
            task_id = task_id.to(device)
            cam_id = cam_id.to(device)

            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                z_task, z_nuis = model(imgs)
                z_task32 = z_task.float()
                z_nuis32 = z_nuis.float()
                # project text in fp32 for numerical stability
                z_text = model.project_text(text_protos_dev.float())
                L_task = supcon_loss(z_task32, task_id, temperature)
                L_nuis = supcon_loss(z_nuis32, cam_id, temperature)
                L_orth = ortho_loss(z_task32, z_nuis32)
                L_vl = alignment_infonce_loss(z_task32, task_id, z_text, temperature)
                loss = L_task + lambda_nuis * L_nuis + lambda_ortho * L_orth + lambda_vl * L_vl

            if is_train:
                (loss / accum_steps).backward()
                if (step + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            sums["loss"] += loss.item()
            sums["L_task"] += L_task.item()
            sums["L_nuis"] += L_nuis.item()
            sums["L_orth"] += L_orth.item()
            sums["L_vl"] += L_vl.item()
            n_steps += 1

            if is_train and log_every and step % log_every == 0:
                msg = {"epoch": epoch, "step": step, "phase": "train",
                       "loss": loss.item(), "L_task": L_task.item(),
                       "L_nuis": L_nuis.item(), "L_orth": L_orth.item(),
                       "L_vl": L_vl.item()}
                print(f"  ep{epoch:02d} step{step:03d}  "
                      f"loss={loss.item():.4f}  task={L_task.item():.4f}  "
                      f"nuis={L_nuis.item():.4f}  vl={L_vl.item():.4f}")
                if log_fp is not None:
                    log_fp.write(json.dumps(msg) + "\n")
                    log_fp.flush()
    return {k: v / max(n_steps, 1) for k, v in sums.items()}


def compute_embeddings_unfreeze(model, loader, device, autocast_dtype):
    model.eval()
    Zt_list, Y_list, C_list = [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs, task_id, cam_id, _, _ = batch
            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                zt, _ = model(imgs)
            Zt_list.append(zt.float().cpu())
            Y_list.append(task_id)
            C_list.append(cam_id)
    return torch.cat(Zt_list, dim=0), torch.cat(Y_list, dim=0), torch.cat(C_list, dim=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patch-h5-root", type=Path, default=Path("data/patch_embeddings"))
    p.add_argument("--clip-root", type=Path, default=Path("data/robocasa_clips"))
    p.add_argument("--unified-meta", type=Path,
                   default=Path("data/patch_embeddings/_unified_subtasks.json"))
    p.add_argument("--cameras", nargs="+", default=None)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--batch-size", type=int, default=39)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-lr-scale", type=float, default=10.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--num-queries", type=int, default=4)
    p.add_argument("--d-task", type=int, default=128)
    p.add_argument("--d-nuis", type=int, default=64)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--unfreeze-last-n", type=int, default=2)
    p.add_argument("--text-proj-hidden", type=int, default=256)
    p.add_argument("--lambda-nuis", type=float, default=1.0)
    p.add_argument("--lambda-ortho", type=float, default=5e-3)
    p.add_argument("--lambda-vl", type=float, default=1.0)
    p.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=5)
    p.add_argument("--probe-every", type=int, default=1)
    p.add_argument("--probe-k", type=int, default=10)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--no-grad-checkpoint", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("runs") / f"exp2_unfreeze_vl_last{args.unfreeze_last_n}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] out_dir={out_dir}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    autocast_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    unified = json.loads(args.unified_meta.read_text())
    unified_mapping: dict[str, int] = unified["mapping"]
    num_subtasks: int = int(unified["num_subtasks"])
    subtask_id_to_label = {int(k): v for k, v in unified["subtask_id_to_label"].items()}
    print(f"[INFO] unified subtasks: {num_subtasks}")

    tasks_from_mapping = sorted({k.split("/")[0] for k in unified_mapping.keys()})
    h5_paths = [args.patch_h5_root / f"{t}.hdf5" for t in tasks_from_mapping]
    for p in h5_paths:
        if not p.exists():
            raise SystemExit(f"HDF5 없음: {p}")

    split_rng = np.random.default_rng(args.seed)
    train_idx, test_idx, cameras, task_order = build_cross_task_indices(
        h5_paths, args.patch_h5_root, unified_mapping,
        args.train_ratio, args.cameras, split_rng,
    )
    num_cameras = len(cameras)

    # ---- text encoding ----
    # extract instructions per sub-task (after " : " separator if present)
    texts = []
    for sid in range(num_subtasks):
        label = subtask_id_to_label[sid]
        # strip "TaskName : " prefix if present
        if " : " in label:
            text = label.split(" : ", 1)[1]
        else:
            text = label
        texts.append(text)
    print(f"[INFO] encoding {len(texts)} sub-task instructions with CLIP...")
    text_protos, clip_info = encode_texts_with_clip(
        texts, model_id=args.clip_model_id, device=device,
    )
    print(f"[INFO] text_protos shape={tuple(text_protos.shape)}")
    torch.save(text_protos, out_dir / "text_protos.pt")

    train_ds = RawClipDataset(train_idx, args.clip_root)
    test_ds = RawClipDataset(test_idx, args.clip_root)

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

    model = UnfreezeVLModel(
        text_dim=clip_info["dim"],
        text_proj_hidden=args.text_proj_hidden,
        unfreeze_last_n=args.unfreeze_last_n,
        num_queries=args.num_queries,
        d_task=args.d_task, d_nuis=args.d_nuis,
        hidden=args.hidden, num_heads=args.num_heads,
        dropout=args.dropout,
        grad_checkpoint=not args.no_grad_checkpoint,
    ).to(device)

    backbone_params = [p for p in model.visual.backbone.parameters() if p.requires_grad]
    head_params = (list(model.visual.pool.parameters())
                   + list(model.visual.head_task.parameters())
                   + list(model.visual.head_nuis.parameters())
                   + list(model.text_proj_net.parameters()))
    n_bb = sum(p.numel() for p in backbone_params)
    n_hd = sum(p.numel() for p in head_params)
    print(f"[INFO] trainable backbone params={n_bb:,}  head+text params={n_hd:,}")

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * args.head_lr_scale},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs)

    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config = {
        "experiment": f"exp2_unfreeze_vl_last{args.unfreeze_last_n}_alignment_infonce",
        "args": args_dict,
        "tasks": task_order,
        "num_subtasks": num_subtasks,
        "subtask_id_to_label": subtask_id_to_label,
        "unified_mapping": unified_mapping,
        "cameras": cameras,
        "h5_paths": [str(p) for p in h5_paths],
        "d_model": model.visual.d_model,
        "model_config": model.config,
        "text_model_id": clip_info["model_id"],
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    log_fp = open(out_dir / "log.jsonl", "w")

    best_test_loss = float("inf")
    best_knn = 0.0
    for epoch in range(args.epochs):
        print(f"\n=== epoch {epoch:02d}/{args.epochs}  "
              f"lr_bb={optimizer.param_groups[0]['lr']:.2e}  "
              f"lr_hd={optimizer.param_groups[1]['lr']:.2e} ===")
        train_stats = run_epoch_unfreeze_vl(
            model, text_protos, train_loader, optimizer, device,
            temperature=args.temperature,
            lambda_nuis=args.lambda_nuis,
            lambda_ortho=args.lambda_ortho,
            lambda_vl=args.lambda_vl,
            autocast_dtype=autocast_dtype, accum_steps=args.accum_steps,
            log_every=args.log_every, epoch=epoch, log_fp=log_fp,
        )
        test_stats = run_epoch_unfreeze_vl(
            model, text_protos, test_loader, None, device,
            temperature=args.temperature,
            lambda_nuis=args.lambda_nuis,
            lambda_ortho=args.lambda_ortho,
            lambda_vl=args.lambda_vl,
            autocast_dtype=autocast_dtype,
        )
        scheduler.step()

        probe_stats = None
        if args.probe_every and (epoch + 1) % args.probe_every == 0:
            Zt_tr, Y_tr, _ = compute_embeddings_unfreeze(model, probe_train_loader, device, autocast_dtype)
            Zt_te, Y_te, _ = compute_embeddings_unfreeze(model, test_loader, device, autocast_dtype)
            knn_acc = knn_accuracy(Zt_tr.to(device), Y_tr.to(device),
                                    Zt_te.to(device), Y_te.to(device), args.probe_k)
            probe_stats = {"knn_acc_subtask": float(knn_acc)}
            if knn_acc > best_knn:
                best_knn = knn_acc

        row = {"epoch": epoch, "phase": "epoch_summary",
               "train": train_stats, "test": test_stats, "probe": probe_stats,
               "lr_bb": optimizer.param_groups[0]["lr"],
               "lr_hd": optimizer.param_groups[1]["lr"]}
        log_fp.write(json.dumps(row) + "\n")
        log_fp.flush()

        print(f"  [train] loss={train_stats['loss']:.4f}  "
              f"task={train_stats['L_task']:.4f}  nuis={train_stats['L_nuis']:.4f}  "
              f"orth={train_stats['L_orth']:.4f}  vl={train_stats['L_vl']:.4f}")
        print(f"  [test]  loss={test_stats['loss']:.4f}  "
              f"task={test_stats['L_task']:.4f}  vl={test_stats['L_vl']:.4f}")
        if probe_stats is not None:
            print(f"  [probe] subtask@z_t kNN({args.probe_k}) = {probe_stats['knn_acc_subtask']:.3f}")

        if test_stats["loss"] < best_test_loss:
            best_test_loss = test_stats["loss"]
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "test_stats": test_stats, "config": config},
                       out_dir / "best.pt")
        if (epoch + 1) % args.ckpt_every == 0 or epoch + 1 == args.epochs:
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "test_stats": test_stats, "config": config},
                       out_dir / f"ckpt_epoch_{epoch:02d}.pt")

    log_fp.close()
    print(f"\n[done] best test loss = {best_test_loss:.4f}  best knn = {best_knn:.4f}  out_dir={out_dir}")


if __name__ == "__main__":
    main()
