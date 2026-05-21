"""P1 — DINOv3 ViT-S/16 last-N blocks unfreeze + cross-task 13-class contrastive.

§3.2 진단으로 frozen DINOv3 patch token 의 representation ceiling 이 ~0.7 임을
확인 → last 2 transformer block 만 trainable 하게 풀어 ceiling 돌파 시도.

기존 contrastive_train_cross_task.py 와 차이:
  - PatchClipDataset (cached .hdf5) → RawClipDataset (PNG load)
  - DisentangleModel (pool+head) → UnfreezeDinoModel (DINOv3 + pool + head)
  - DINOv3 last N block 만 requires_grad=True, 나머지 frozen
  - bf16 autocast + last-block gradient checkpointing

Memory 추정 (bs=32):
  raw image: 32 × 4 × 224×224×3 = 19 MB
  DINOv3 fwd activations (no grad, frozen): ~150 MB peak
  unfrozen last 2 blocks activation+grad: ~80 MB
  total ≈ 5~6 GB on 16GB GPU → OK
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

from contrastive_train import (
    AttentionPool,
    BalancedBatchSampler,
    ProjectionHead,
    SampleIndex,
    knn_accuracy,
    make_scheduler,
    ortho_loss,
    supcon_loss,
)
from contrastive_train_cross_task import build_cross_task_indices


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class RawClipDataset(Dataset):
    """clip_root/{task}/{camera}/ep{ep:03d}_clip{clip_id:02d}_f{f:02d}.png 로드."""

    def __init__(self, indices: list[SampleIndex], clip_root: Path, n_frames: int = 4):
        self.indices = indices
        self.clip_root = clip_root
        self.n_frames = n_frames

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        s = self.indices[i]
        task = Path(s.h5_path).stem
        d = self.clip_root / task / s.camera
        frames = []
        for f in range(self.n_frames):
            p = d / f"ep{s.ep_idx:03d}_clip{s.clip_id:02d}_f{f:02d}.png"
            img = Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR)
            frames.append(np.asarray(img, dtype=np.uint8))
        # stack: (n, H, W, 3) → (n, 3, H, W) float32 / 255, then normalize
        arr = np.stack(frames, axis=0)
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
        t = (t - IMAGENET_MEAN.squeeze(0)) / IMAGENET_STD.squeeze(0)
        return t, s.task_id, s.camera_id, s.ep_idx, s.clip_id


class UnfreezeDinoModel(nn.Module):
    """DINOv3 backbone + AttentionPool + 2 heads. Last N block 만 trainable."""

    def __init__(
        self,
        model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        unfreeze_last_n: int = 2,
        num_queries: int = 4,
        d_task: int = 128,
        d_nuis: int = 64,
        hidden: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        grad_checkpoint: bool = True,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_id)
        d_model = int(self.backbone.config.hidden_size)
        self.num_register = int(self.backbone.config.num_register_tokens)
        self.unfreeze_last_n = unfreeze_last_n

        # freeze everything first
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        # unfreeze last N blocks
        n_blocks = len(self.backbone.model.layer)
        unfreeze_start = max(0, n_blocks - unfreeze_last_n)
        for i in range(unfreeze_start, n_blocks):
            for p in self.backbone.model.layer[i].parameters():
                p.requires_grad_(True)
        # also unfreeze final norm
        for p in self.backbone.norm.parameters():
            p.requires_grad_(True)

        if grad_checkpoint:
            self.backbone.gradient_checkpointing_enable()

        self.pool = AttentionPool(d_model, num_queries, num_heads, dropout)
        in_dim = num_queries * d_model
        self.head_task = ProjectionHead(in_dim, hidden, d_task)
        self.head_nuis = ProjectionHead(in_dim, hidden, d_nuis)
        self.d_model = d_model
        self.num_queries = num_queries
        self.config = dict(
            model_id=model_id,
            unfreeze_last_n=unfreeze_last_n,
            d_model=d_model,
            num_queries=num_queries,
            d_task=d_task,
            d_nuis=d_nuis,
            hidden=hidden,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # images: (B, n_frames, 3, H, W) → flatten to (B*n, 3, H, W)
        B, N, C, H, W = images.shape
        x = images.view(B * N, C, H, W)
        out = self.backbone(pixel_values=x)
        last = out.last_hidden_state                  # (B*N, 1+R+P, D)
        patches = last[:, 1 + self.num_register :]    # (B*N, P, D)  P=196
        # reshape to (B, N*P, D)
        P = patches.shape[1]
        patches = patches.reshape(B, N * P, self.d_model)
        h = self.pool(patches).flatten(1)
        return self.head_task(h), self.head_nuis(h)


def knn_predict(Z_tr: torch.Tensor, y_tr: torch.Tensor, Z_te: torch.Tensor, k: int):
    """cosine kNN top-k vote — fully on GPU; returns predicted labels."""
    Z_tr = F.normalize(Z_tr, dim=1)
    Z_te = F.normalize(Z_te, dim=1)
    sim = Z_te @ Z_tr.T
    top = sim.topk(k=k, dim=1).indices
    nn_lbl = y_tr[top]
    preds = []
    for row in nn_lbl:
        vals, counts = torch.unique(row, return_counts=True)
        preds.append(int(vals[counts.argmax()]))
    return torch.tensor(preds, device=Z_te.device)


def compute_embeddings(model, loader, device, autocast_dtype):
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
    Zt = torch.cat(Zt_list, dim=0)
    Y = torch.cat(Y_list, dim=0)
    C = torch.cat(C_list, dim=0)
    return Zt, Y, C


def run_epoch_unfreeze(
    model, loader, optimizer, device, *,
    temperature, lambda_nuis, lambda_ortho,
    autocast_dtype, accum_steps=1, scaler=None,
    log_every=0, epoch=0, log_fp=None,
):
    is_train = optimizer is not None
    model.train(is_train)
    sums = {"loss": 0.0, "L_task": 0.0, "L_nuis": 0.0, "L_orth": 0.0}
    n_steps = 0
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
                L_task = supcon_loss(z_task.float(), task_id, temperature)
                L_nuis = supcon_loss(z_nuis.float(), cam_id, temperature)
                L_orth = ortho_loss(z_task.float(), z_nuis.float())
                loss = L_task + lambda_nuis * L_nuis + lambda_ortho * L_orth

            if is_train:
                (loss / accum_steps).backward()
                if (step + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            sums["loss"] += loss.item()
            sums["L_task"] += L_task.item()
            sums["L_nuis"] += L_nuis.item()
            sums["L_orth"] += L_orth.item()
            n_steps += 1

            if is_train and log_every and step % log_every == 0:
                msg = {"epoch": epoch, "step": step, "phase": "train",
                       "loss": loss.item(), "L_task": L_task.item(),
                       "L_nuis": L_nuis.item(), "L_orth": L_orth.item()}
                print(
                    f"  ep{epoch:02d} step{step:03d}  "
                    f"loss={loss.item():.4f}  task={L_task.item():.4f}  "
                    f"nuis={L_nuis.item():.4f}  orth={L_orth.item():.4f}"
                )
                if log_fp is not None:
                    log_fp.write(json.dumps(msg) + "\n")
                    log_fp.flush()
    return {k: v / max(n_steps, 1) for k, v in sums.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patch-h5-root", type=Path,
                   default=Path("data/patch_embeddings"))
    p.add_argument("--clip-root", type=Path,
                   default=Path("data/robocasa_clips"))
    p.add_argument("--unified-meta", type=Path,
                   default=Path("data/patch_embeddings/_unified_subtasks.json"))
    p.add_argument("--cameras", nargs="+", default=None)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--batch-size", type=int, default=39,
                   help="13 sub-task × 3 cam = 39 cells; per-cell=1")
    p.add_argument("--accum-steps", type=int, default=3,
                   help="effective batch = bs × accum")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="DINOv3 layers; head LR auto-scaled 10x")
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
    p.add_argument("--lambda-nuis", type=float, default=1.0)
    p.add_argument("--lambda-ortho", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=5)
    p.add_argument("--probe-every", type=int, default=1)
    p.add_argument("--probe-k", type=int, default=10)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--no-grad-checkpoint", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("runs") / f"exp2_unfreeze_last{args.unfreeze_last_n}_{ts}"
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

    train_ds = RawClipDataset(train_idx, args.clip_root)
    test_ds = RawClipDataset(test_idx, args.clip_root)

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

    model = UnfreezeDinoModel(
        unfreeze_last_n=args.unfreeze_last_n,
        num_queries=args.num_queries,
        d_task=args.d_task, d_nuis=args.d_nuis,
        hidden=args.hidden, num_heads=args.num_heads,
        dropout=args.dropout,
        grad_checkpoint=not args.no_grad_checkpoint,
    ).to(device)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.pool.parameters()) + list(model.head_task.parameters()) + list(model.head_nuis.parameters())
    n_bb = sum(p.numel() for p in backbone_params)
    n_hd = sum(p.numel() for p in head_params)
    print(f"[INFO] trainable backbone params={n_bb:,}  head params={n_hd:,}")

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
        "experiment": f"exp2_unfreeze_last{args.unfreeze_last_n}",
        "args": args_dict,
        "tasks": task_order,
        "num_subtasks": num_subtasks,
        "subtask_id_to_label": subtask_id_to_label,
        "unified_mapping": unified_mapping,
        "cameras": cameras,
        "h5_paths": [str(p) for p in h5_paths],
        "d_model": model.d_model,
        "model_config": model.config,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    log_fp = open(out_dir / "log.jsonl", "w")

    best_test_loss = float("inf")
    for epoch in range(args.epochs):
        print(
            f"\n=== epoch {epoch:02d}/{args.epochs}  "
            f"lr_bb={optimizer.param_groups[0]['lr']:.2e}  "
            f"lr_hd={optimizer.param_groups[1]['lr']:.2e} ==="
        )
        train_stats = run_epoch_unfreeze(
            model, train_loader, optimizer, device,
            temperature=args.temperature,
            lambda_nuis=args.lambda_nuis, lambda_ortho=args.lambda_ortho,
            autocast_dtype=autocast_dtype, accum_steps=args.accum_steps,
            log_every=args.log_every, epoch=epoch, log_fp=log_fp,
        )
        test_stats = run_epoch_unfreeze(
            model, test_loader, None, device,
            temperature=args.temperature,
            lambda_nuis=args.lambda_nuis, lambda_ortho=args.lambda_ortho,
            autocast_dtype=autocast_dtype,
        )
        scheduler.step()

        probe_stats = None
        if args.probe_every and (epoch + 1) % args.probe_every == 0:
            Zt_tr, Y_tr, _ = compute_embeddings(model, probe_train_loader, device, autocast_dtype)
            Zt_te, Y_te, _ = compute_embeddings(model, test_loader, device, autocast_dtype)
            knn_acc = knn_accuracy(Zt_tr.to(device), Y_tr.to(device),
                                    Zt_te.to(device), Y_te.to(device), args.probe_k)
            probe_stats = {"knn_acc_subtask": float(knn_acc)}

        row = {"epoch": epoch, "phase": "epoch_summary",
               "train": train_stats, "test": test_stats, "probe": probe_stats,
               "lr_bb": optimizer.param_groups[0]["lr"],
               "lr_hd": optimizer.param_groups[1]["lr"]}
        log_fp.write(json.dumps(row) + "\n")
        log_fp.flush()

        print(f"  [train] loss={train_stats['loss']:.4f}  "
              f"task={train_stats['L_task']:.4f}  nuis={train_stats['L_nuis']:.4f}  "
              f"orth={train_stats['L_orth']:.4f}")
        print(f"  [test]  loss={test_stats['loss']:.4f}  "
              f"task={test_stats['L_task']:.4f}  nuis={test_stats['L_nuis']:.4f}")
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
    print(f"\n[done] best test loss = {best_test_loss:.4f}  out_dir={out_dir}")


if __name__ == "__main__":
    main()
