"""Cross-task fusion: CLIP text token + DINOv3 patch tokens → AttentionPool.

Exp 2 의 변형. text 가 loss 가 아니라 input 으로 들어감 (early fusion):
  patches (B, N, 384) + text_token (B, 1, 384) → (B, N+1, 384) → AttentionPool K=4 query.

핵심 설계:
  - CLIP text proto 는 frozen, learned text_proj 로 512→384 사영
  - text_dropout p (default 0.5): 학습 중 일부 sample 의 text 를 학습 가능한 null_token 으로 대체
    → vision side 가 text-only 정답 leak 에 의존하지 못하게
  - Probe 시:
      * with-text  : 각 sample 의 정답 text 를 함께 넣고 평가 (text leak 영향 그대로)
      * no-text    : null_token 으로 평가 (순수 visual representation)
        no-text 가 vision-only baseline 보다 좋다면 fusion training 이 visual 도 개선한 것

사용:
  python script/contrastive_train_cross_task_fusion.py --epochs 60
  python script/contrastive_train_cross_task_fusion.py --text-dropout 0.0   # 비교용
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from contrastive_train import (
    AttentionPool,
    BalancedBatchSampler,
    PatchClipDataset,
    ProjectionHead,
    knn_accuracy,
    make_scheduler,
    ortho_loss,
    supcon_loss,
)
from contrastive_train_cross_task import build_cross_task_indices
from contrastive_train_cross_task_vl import build_subtask_instruction_texts
from contrastive_train_subtask_vl import encode_texts_with_clip


# ---------------------------------------------------------------------------
# Fusion model
# ---------------------------------------------------------------------------


class FusionDisentangleModel(nn.Module):
    """patches + text-token concat → AttentionPool → 2 heads.

    text_emb_input shape: (B, text_dim) — already L2-normalized CLIP output.
    """

    def __init__(
        self,
        d_model: int = 384,
        text_dim: int = 512,
        num_queries: int = 4,
        d_task: int = 128,
        d_nuis: int = 64,
        hidden: int = 512,
        text_proj_hidden: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_proj_net = nn.Sequential(
            nn.Linear(text_dim, text_proj_hidden),
            nn.GELU(),
            nn.Linear(text_proj_hidden, d_model),
            nn.LayerNorm(d_model),
        )
        # 학습 가능한 NULL token (text 가 없을 때 자리 채움)
        self.null_token = nn.Parameter(torch.randn(d_model) * 0.02)

        self.pool = AttentionPool(d_model, num_queries, num_heads, dropout)
        in_dim = num_queries * d_model
        self.head_task = ProjectionHead(in_dim, hidden, d_task)
        self.head_nuis = ProjectionHead(in_dim, hidden, d_nuis)

        self.config = dict(
            d_model=d_model, text_dim=text_dim, num_queries=num_queries,
            d_task=d_task, d_nuis=d_nuis, hidden=hidden,
            text_proj_hidden=text_proj_hidden,
            num_heads=num_heads, dropout=dropout,
        )

    def forward(
        self,
        patches: torch.Tensor,        # (B, N, d_model)
        text_emb: torch.Tensor | None,  # (B, text_dim) or None → all null
        text_mask: torch.Tensor | None = None,  # (B,) bool, True = use text, False = use null
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = patches.shape[0]
        device = patches.device

        if text_emb is None:
            text_tok = self.null_token.unsqueeze(0).expand(B, -1)
        else:
            proj = self.text_proj_net(text_emb)              # (B, d_model)
            if text_mask is not None:
                null_b = self.null_token.unsqueeze(0).expand(B, -1)
                proj = torch.where(text_mask.unsqueeze(-1), proj, null_b)
            text_tok = proj

        x = torch.cat([patches, text_tok.unsqueeze(1)], dim=1)  # (B, N+1, d_model)
        h = self.pool(x).flatten(1)                            # (B, K * d_model)
        return self.head_task(h), self.head_nuis(h)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def run_epoch_fusion(
    model: FusionDisentangleModel,
    text_protos: torch.Tensor,         # (n_sub, text_dim)
    loader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    *,
    temperature: float,
    lambda_nuis: float,
    lambda_ortho: float,
    text_dropout: float,
    log_every: int = 0,
    epoch: int = 0,
    log_fp=None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    sums = {"loss": 0.0, "L_task": 0.0, "L_nuis": 0.0, "L_orth": 0.0}
    n_steps = 0

    text_protos_dev = text_protos.to(device)
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for step, batch in enumerate(loader):
            patches, sub_task, cam_id, _, _ = batch
            patches = patches.to(device, non_blocking=True)
            sub_task = sub_task.to(device)
            cam_id = cam_id.to(device)

            text_b = text_protos_dev[sub_task]    # (B, text_dim)
            if is_train and text_dropout > 0:
                keep = torch.rand(text_b.shape[0], device=device) > text_dropout
            else:
                keep = torch.ones(text_b.shape[0], device=device, dtype=torch.bool)

            z_task, z_nuis = model(patches, text_b, text_mask=keep)

            L_task = supcon_loss(z_task, sub_task, temperature)
            L_nuis = supcon_loss(z_nuis, cam_id, temperature)
            L_orth = ortho_loss(z_task, z_nuis)
            loss = L_task + lambda_nuis * L_nuis + lambda_ortho * L_orth

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            for k, v in zip(
                ["loss", "L_task", "L_nuis", "L_orth"],
                [loss, L_task, L_nuis, L_orth],
            ):
                sums[k] += float(v.item())
            n_steps += 1

            if is_train and log_every and step % log_every == 0:
                print(
                    f"  ep{epoch:02d} step{step:03d}  "
                    f"loss={loss.item():.4f}  task={L_task.item():.4f}  "
                    f"nuis={L_nuis.item():.4f}  orth={L_orth.item():.4f}  "
                    f"keep_frac={keep.float().mean().item():.2f}"
                )
                if log_fp is not None:
                    log_fp.write(json.dumps({
                        "epoch": epoch, "step": step, "phase": "train",
                        "loss": loss.item(), "L_task": L_task.item(),
                        "L_nuis": L_nuis.item(), "L_orth": L_orth.item(),
                    }) + "\n")
                    log_fp.flush()
    return {k: v / max(n_steps, 1) for k, v in sums.items()}


@torch.no_grad()
def compute_embeddings_fusion(
    model: FusionDisentangleModel,
    text_protos: torch.Tensor,
    loader,
    device: str,
    *,
    text_mode: str,    # "with_text" or "no_text"
):
    model.eval()
    Zt, Zn, Yt, Yc = [], [], [], []
    text_protos_dev = text_protos.to(device)
    for batch in loader:
        patches, sub_task, cam_id, _, _ = batch
        patches = patches.to(device, non_blocking=True)
        sub_task_d = sub_task.to(device)

        if text_mode == "with_text":
            text_b = text_protos_dev[sub_task_d]
            mask = torch.ones(patches.shape[0], device=device, dtype=torch.bool)
        elif text_mode == "no_text":
            text_b = None
            mask = None
        else:
            raise ValueError(text_mode)

        z_t, z_n = model(patches, text_b, text_mask=mask)
        Zt.append(z_t.cpu())
        Zn.append(z_n.cpu())
        Yt.append(sub_task if isinstance(sub_task, torch.Tensor) else torch.as_tensor(sub_task))
        Yc.append(cam_id if isinstance(cam_id, torch.Tensor) else torch.as_tensor(cam_id))
    return torch.cat(Zt), torch.cat(Zn), torch.cat(Yt), torch.cat(Yc)


@torch.no_grad()
def run_probes_fusion(model, text_protos, train_loader, test_loader, device, k=10):
    probes = {}
    for mode in ["with_text", "no_text"]:
        zt_tr, zn_tr, yt_tr, yc_tr = compute_embeddings_fusion(
            model, text_protos, train_loader, device, text_mode=mode
        )
        zt_te, zn_te, yt_te, yc_te = compute_embeddings_fusion(
            model, text_protos, test_loader, device, text_mode=mode
        )
        probes[mode] = {
            "task_on_z_task": knn_accuracy(zt_tr, yt_tr, zt_te, yt_te, k),
            "cam_on_z_task":  knn_accuracy(zt_tr, yc_tr, zt_te, yc_te, k),
            "task_on_z_nuis": knn_accuracy(zn_tr, yt_tr, zn_te, yt_te, k),
            "cam_on_z_nuis":  knn_accuracy(zn_tr, yc_tr, zn_te, yc_te, k),
        }
    return probes


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patch-h5-root", type=Path, default=Path("data/patch_embeddings"))
    p.add_argument("--unified-meta", type=Path,
                   default=Path("data/patch_embeddings/_unified_subtasks.json"))
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
    p.add_argument("--text-proj-hidden", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lambda-nuis", type=float, default=1.0)
    p.add_argument("--lambda-ortho", type=float, default=5e-3)
    p.add_argument("--text-dropout", type=float, default=0.5,
                   help="확률 p 로 학습 시 text 토큰을 null 로 대체 (0=항상 text, 1=항상 null)")
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
    suffix = f"fusion_drop{args.text_dropout:.2f}".replace(".", "")
    out_dir = args.out_dir or Path("runs") / f"exp2_cross_task_{suffix}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] out_dir={out_dir}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    unified = json.loads(args.unified_meta.read_text())
    unified_mapping = unified["mapping"]
    num_subtasks = int(unified["num_subtasks"])
    subtask_id_to_label = {int(k): v for k, v in unified["subtask_id_to_label"].items()}

    sid_to_text = build_subtask_instruction_texts(unified_mapping, args.patch_h5_root)
    instruction_strs = [sid_to_text[i] for i in range(num_subtasks)]
    print(f"[INFO] encoding {num_subtasks} sub-task texts with CLIP {args.clip_model}")
    text_protos, clip_info = encode_texts_with_clip(
        instruction_strs, model_id=args.clip_model, device=device
    )
    print(f"[INFO] text_protos shape={tuple(text_protos.shape)}")

    tasks_from_mapping = sorted({k.split("/")[0] for k in unified_mapping.keys()})
    h5_paths = [args.patch_h5_root / f"{t}.hdf5" for t in tasks_from_mapping]
    missing = [p for p in h5_paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing HDF5: {missing}")

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

    model = FusionDisentangleModel(
        d_model=d_model,
        text_dim=clip_info["dim"],
        num_queries=args.num_queries,
        d_task=args.d_task, d_nuis=args.d_nuis,
        hidden=args.hidden, text_proj_hidden=args.text_proj_hidden,
        num_heads=args.num_heads, dropout=args.dropout,
    ).to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] trainable params={n_train:,}  device={device}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs)

    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config = {
        "experiment": "exp2_cross_task_fusion",
        "args": args_dict,
        "clip_model": args.clip_model,
        "tasks": task_order,
        "num_subtasks": num_subtasks,
        "subtask_id_to_label": {int(k): v for k, v in subtask_id_to_label.items()},
        "subtask_id_to_text": sid_to_text,
        "unified_mapping": unified_mapping,
        "cameras": cameras,
        "h5_paths": [str(p) for p in h5_paths],
        "d_model": d_model,
        "text_dim": clip_info["dim"],
        "model_config": model.config,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    torch.save(text_protos, out_dir / "text_protos.pt")
    log_fp = open(out_dir / "log.jsonl", "w")

    best_test_loss = float("inf")
    for epoch in range(args.epochs):
        print(f"\n=== epoch {epoch:02d}/{args.epochs}  lr={optimizer.param_groups[0]['lr']:.2e} ===")
        train_stats = run_epoch_fusion(
            model, text_protos, train_loader, optimizer, device,
            temperature=args.temperature,
            lambda_nuis=args.lambda_nuis, lambda_ortho=args.lambda_ortho,
            text_dropout=args.text_dropout,
            log_every=args.log_every, epoch=epoch, log_fp=log_fp,
        )
        test_stats = run_epoch_fusion(
            model, text_protos, test_loader, None, device,
            temperature=args.temperature,
            lambda_nuis=args.lambda_nuis, lambda_ortho=args.lambda_ortho,
            text_dropout=0.0,
        )
        scheduler.step()

        probe_stats = None
        if args.probe_every and (epoch + 1) % args.probe_every == 0:
            probe_stats = run_probes_fusion(
                model, text_protos, probe_train_loader, test_loader, device, k=args.probe_k
            )

        log_fp.write(json.dumps({
            "epoch": epoch, "phase": "epoch_summary",
            "train": train_stats, "test": test_stats, "probe": probe_stats,
            "lr": optimizer.param_groups[0]["lr"],
        }) + "\n")
        log_fp.flush()

        print(f"  [train] loss={train_stats['loss']:.4f}  task={train_stats['L_task']:.4f}  "
              f"nuis={train_stats['L_nuis']:.4f}  orth={train_stats['L_orth']:.4f}")
        print(f"  [test]  loss={test_stats['loss']:.4f}  task={test_stats['L_task']:.4f}")
        if probe_stats:
            wt = probe_stats["with_text"]; nt = probe_stats["no_text"]
            print(f"  [probe with_text] subtask@z_t={wt['task_on_z_task']:.3f}  "
                  f"cam@z_t={wt['cam_on_z_task']:.3f}  cam@z_n={wt['cam_on_z_nuis']:.3f}")
            print(f"  [probe no_text  ] subtask@z_t={nt['task_on_z_task']:.3f}  "
                  f"cam@z_t={nt['cam_on_z_task']:.3f}  cam@z_n={nt['cam_on_z_nuis']:.3f}")

        if test_stats["loss"] < best_test_loss:
            best_test_loss = test_stats["loss"]
            torch.save({"model": model.state_dict(),
                        "epoch": epoch, "test_stats": test_stats, "config": config},
                       out_dir / "best.pt")
        if (epoch + 1) % args.ckpt_every == 0 or epoch + 1 == args.epochs:
            torch.save({"model": model.state_dict(),
                        "epoch": epoch, "test_stats": test_stats, "config": config},
                       out_dir / f"ckpt_epoch_{epoch:02d}.pt")

    log_fp.close()
    print(f"\n[done] best test loss = {best_test_loss:.4f}  out_dir={out_dir}")


if __name__ == "__main__":
    main()
