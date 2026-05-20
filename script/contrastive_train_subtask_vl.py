"""Within-task sub-task contrastive with vision-language alignment (Exp 1b).

기존 contrastive_train_subtask.py 위에 **CLIP ViT-B/32 text encoder (frozen)** 를
추가해서 visual z_task 와 instruction text embedding 을 정렬하는 학습.

두 loss 모드 (--vl-loss):
    alignment_infonce
        각 sample 의 z_task 를 sub-task text prototype 들에 대해 softmax 분류.
        L_vl = CE(z_task @ z_text_protos.T / τ , sub_task_id)
        효과: text 가 fixed classifier head 역할 (semantic-aware classification).
    anchor_regression
        각 sample 의 z_task 를 그 sub-task 의 text anchor 로 cosine pull.
        L_vl = mean(1 - cos(z_task, z_text[sub_task_id]))
        효과: text 가 fixed anchor (negative push 없음, attractive only).

다른 loss 는 그대로 유지:
    L_task: SupCon(z_task, sub-task id)        ← 기존 visual contrastive
    L_nuis: SupCon(z_nuis, camera id)
    L_orth: orthogonality(z_task, z_nuis)
    total = L_task + λ_n·L_nuis + λ_o·L_orth + λ_vl·L_vl

사용 예:
    python script/contrastive_train_subtask_vl.py --task CloseDrawer --vl-loss alignment_infonce
    python script/contrastive_train_subtask_vl.py --task AdjustWaterTemperature --vl-loss anchor_regression
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
from transformers import CLIPTextModel, CLIPTokenizer

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
from contrastive_train_subtask import (
    build_subtask_indices,
    load_instruction_meta,
)


# ---------------------------------------------------------------------------
# CLIP text encoding (frozen, one-time)
# ---------------------------------------------------------------------------


def encode_texts_with_clip(
    texts: list[str],
    model_id: str = "openai/clip-vit-base-patch32",
    device: str = "cuda",
) -> tuple[torch.Tensor, dict]:
    """CLIP text encoder 로 인코딩. Returns (N, 512) L2-norm."""
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    encoder = CLIPTextModel.from_pretrained(model_id).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        toks = tokenizer(
            texts, padding=True, return_tensors="pt", truncation=True
        ).to(device)
        out = encoder(**toks)
        embs = out.pooler_output       # (N, 512)
        embs = F.normalize(embs, dim=-1)
    return embs.cpu(), {"model_id": model_id, "dim": int(embs.shape[1])}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class VLModel(nn.Module):
    """기존 DisentangleModel + text projection head.

    visual: patches → (z_task, z_nuis)
    text_proj: pre-computed CLIP embedding (512-d) → z_text (d_task-d) L2-norm
    """

    def __init__(
        self,
        visual_model: DisentangleModel,
        text_dim: int,
        d_task: int,
        hidden: int = 256,
    ):
        super().__init__()
        self.visual = visual_model
        self.text_proj_net = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_task),
            nn.LayerNorm(d_task),
        )

    def forward(self, patches):
        return self.visual(patches)

    def project_text(self, text_emb: torch.Tensor) -> torch.Tensor:
        z = self.text_proj_net(text_emb)
        return F.normalize(z, dim=-1)


# ---------------------------------------------------------------------------
# Vision-Language loss variants
# ---------------------------------------------------------------------------


def alignment_infonce_loss(
    z_task: torch.Tensor,
    sub_task_ids: torch.Tensor,
    text_protos: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """z_task ↔ text prototype 들에 대한 softmax 분류.

    z_task: (B, d) L2-norm, text_protos: (n_sub, d) L2-norm
    """
    logits = z_task @ text_protos.T / temperature      # (B, n_sub)
    return F.cross_entropy(logits, sub_task_ids)


def anchor_regression_loss(
    z_task: torch.Tensor,
    sub_task_ids: torch.Tensor,
    text_protos: torch.Tensor,
) -> torch.Tensor:
    """z_task 를 그 sub-task 의 text anchor 로 cosine pull."""
    target = text_protos[sub_task_ids]                 # (B, d)
    sim = (z_task * target).sum(-1)
    return (1 - sim).mean()


# ---------------------------------------------------------------------------
# Train / eval epoch
# ---------------------------------------------------------------------------


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
            z_text = vl_model.project_text(text_protos_dev)     # (n_sub, d_task)

            L_task = supcon_loss(z_task, task_id, temperature)
            L_nuis = supcon_loss(z_nuis, cam_id, temperature)
            L_orth = ortho_loss(z_task, z_nuis)
            if vl_loss_kind == "alignment_infonce":
                L_vl = alignment_infonce_loss(z_task, task_id, z_text, temperature)
            elif vl_loss_kind == "anchor_regression":
                L_vl = anchor_regression_loss(z_task, task_id, z_text)
            else:
                raise ValueError(f"unknown vl_loss_kind: {vl_loss_kind}")

            loss = (
                L_task
                + lambda_nuis * L_nuis
                + lambda_ortho * L_orth
                + lambda_vl * L_vl
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            sums["loss"] += loss.item()
            sums["L_task"] += L_task.item()
            sums["L_nuis"] += L_nuis.item()
            sums["L_orth"] += L_orth.item()
            sums["L_vl"] += L_vl.item()
            n_steps += 1

            if is_train and log_every and step % log_every == 0:
                msg = {
                    "epoch": epoch, "step": step, "phase": "train",
                    "loss": loss.item(), "L_task": L_task.item(),
                    "L_nuis": L_nuis.item(), "L_orth": L_orth.item(),
                    "L_vl": L_vl.item(),
                }
                print(
                    f"  ep{epoch:02d} step{step:03d}  "
                    f"loss={loss.item():.4f}  task={L_task.item():.4f}  "
                    f"nuis={L_nuis.item():.4f}  vl={L_vl.item():.4f}"
                )
                if log_fp is not None:
                    log_fp.write(json.dumps(msg) + "\n")
                    log_fp.flush()

    return {k: v / max(n_steps, 1) for k, v in sums.items()}


@torch.no_grad()
def run_probes_vl(
    vl_model: VLModel,
    text_protos: torch.Tensor,
    train_loader,
    test_loader,
    device: str,
    k: int = 10,
) -> dict[str, float]:
    """기존 4-way probe + text alignment 정확도."""
    zt_tr, zn_tr, yt_tr, yc_tr = compute_embeddings(vl_model.visual, train_loader, device)
    zt_te, zn_te, yt_te, yc_te = compute_embeddings(vl_model.visual, test_loader, device)

    z_text = vl_model.project_text(text_protos.to(device)).cpu()   # (n_sub, d)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--task", required=True)
    p.add_argument("--patch-h5-root", type=Path, default=Path("data/patch_embeddings"))
    p.add_argument("--instructions-json", type=Path, default=None)
    p.add_argument(
        "--vl-loss",
        choices=["alignment_infonce", "anchor_regression"],
        required=True,
    )
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
    p.add_argument(
        "--freeze-text-proj",
        action="store_true",
        help="text projection head 를 random init 후 동결 (collapse 방지용)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="기본: runs/exp1_{task}_vl_{vl_loss}_{timestamp}",
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
    out_dir = args.out_dir or Path("runs") / f"exp1_{args.task}_vl_{args.vl_loss}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] out_dir={out_dir}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    h5_path = args.patch_h5_root / f"{args.task}.hdf5"
    if not h5_path.exists():
        raise SystemExit(f"HDF5 없음: {h5_path}")
    instr_json = args.instructions_json or (
        args.patch_h5_root / f"{args.task}_instructions.json"
    )
    if not instr_json.exists():
        raise SystemExit(f"instruction JSON 없음: {instr_json}")

    instr_meta = load_instruction_meta(instr_json)
    print(f"[INFO] task={args.task}  h5={h5_path}  instr={instr_json}")
    print(
        f"[INFO] {instr_meta['num_episodes']} episodes, "
        f"{instr_meta['num_unique_instructions']} sub-tasks"
    )

    # ---- Text encoding (frozen, one-time) ----
    n_sub = instr_meta["num_unique_instructions"]
    instruction_strs = [instr_meta["instructions"][str(i)] for i in range(n_sub)]
    print(f"[INFO] encoding {n_sub} instructions with CLIP {args.clip_model}")
    for i, s in enumerate(instruction_strs):
        print(f"         [{i}] {s!r}")
    text_protos, clip_info = encode_texts_with_clip(
        instruction_strs, model_id=args.clip_model, device=device
    )
    print(f"[INFO] text_protos shape={tuple(text_protos.shape)}")

    # ---- Data ----
    split_rng = np.random.default_rng(args.seed)
    train_idx, test_idx, subtask_to_instr, cameras = build_subtask_indices(
        h5_path, instr_meta, args.train_ratio, args.cameras, split_rng
    )
    num_subtasks = n_sub
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
    with h5py.File(h5_path, "r") as f:
        d_model = int(f.attrs["dim"])
    visual_model = DisentangleModel(
        d_model=d_model, num_queries=args.num_queries,
        d_task=args.d_task, d_nuis=args.d_nuis,
        hidden=args.hidden, num_heads=args.num_heads, dropout=args.dropout,
    )
    vl_model = VLModel(
        visual_model,
        text_dim=clip_info["dim"],
        d_task=args.d_task,
        hidden=args.text_proj_hidden,
    ).to(device)

    # F1 처방: text projection 동결
    if args.freeze_text_proj:
        for p in vl_model.text_proj_net.parameters():
            p.requires_grad_(False)
        # 동결 후 두 sub-task text 가 충분히 다른지 sanity log
        with torch.no_grad():
            z_text_check = vl_model.project_text(text_protos.to(device)).cpu()
            if z_text_check.shape[0] >= 2:
                pair_cos = float((z_text_check[0] * z_text_check[1]).sum())
                print(f"[INFO] freeze-text-proj enabled. z_text pair cosine = {pair_cos:.4f}")

    n_params_total = sum(p.numel() for p in vl_model.parameters())
    n_params_train = sum(p.numel() for p in vl_model.parameters() if p.requires_grad)
    print(f"[INFO] trainable params={n_params_train:,} / total {n_params_total:,}  device={device}")

    optimizer = torch.optim.AdamW(
        [p for p in vl_model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs)

    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config = {
        "experiment": "exp1_within_task_visual_language",
        "args": args_dict,
        "task": args.task,
        "vl_loss": args.vl_loss,
        "clip_model": args.clip_model,
        "subtask_to_instruction": {int(k): v for k, v in subtask_to_instr.items()},
        "cameras": cameras,
        "h5_path": str(h5_path),
        "instructions_json": str(instr_json),
        "d_model": d_model,
        "model_config": visual_model.config,
        "text_dim": clip_info["dim"],
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    torch.save(text_protos, out_dir / "text_protos.pt")
    log_fp = open(out_dir / "log.jsonl", "w")

    # ---- Training ----
    best_test_loss = float("inf")
    for epoch in range(args.epochs):
        print(
            f"\n=== epoch {epoch:02d}/{args.epochs}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e} ==="
        )
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

        probe_stats: dict[str, float] | None = None
        if args.probe_every and (epoch + 1) % args.probe_every == 0:
            probe_stats = run_probes_vl(
                vl_model, text_protos, probe_train_loader, test_loader, device, k=args.probe_k
            )

        row = {
            "epoch": epoch, "phase": "epoch_summary",
            "train": train_stats, "test": test_stats, "probe": probe_stats,
            "lr": optimizer.param_groups[0]["lr"],
        }
        log_fp.write(json.dumps(row) + "\n")
        log_fp.flush()

        print(
            f"  [train] loss={train_stats['loss']:.4f}  "
            f"task={train_stats['L_task']:.4f}  nuis={train_stats['L_nuis']:.4f}  "
            f"orth={train_stats['L_orth']:.4f}  vl={train_stats['L_vl']:.4f}"
        )
        print(
            f"  [test]  loss={test_stats['loss']:.4f}  "
            f"task={test_stats['L_task']:.4f}  vl={test_stats['L_vl']:.4f}"
        )
        if probe_stats is not None:
            print(
                f"  [probe] subtask@z_t={probe_stats['task_on_z_task']:.3f}  "
                f"text_align@z_t={probe_stats['text_align_acc']:.3f}  "
                f"cam@z_t={probe_stats['cam_on_z_task']:.3f}  |  "
                f"cam@z_n={probe_stats['cam_on_z_nuis']:.3f}  "
                f"subtask@z_n={probe_stats['task_on_z_nuis']:.3f}"
            )

        if test_stats["loss"] < best_test_loss:
            best_test_loss = test_stats["loss"]
            torch.save(
                {"vl_model": vl_model.state_dict(),
                 "epoch": epoch, "test_stats": test_stats, "config": config},
                out_dir / "best.pt",
            )

        if (epoch + 1) % args.ckpt_every == 0 or epoch + 1 == args.epochs:
            torch.save(
                {"vl_model": vl_model.state_dict(),
                 "epoch": epoch, "test_stats": test_stats, "config": config},
                out_dir / f"ckpt_epoch_{epoch:02d}.pt",
            )

    log_fp.close()
    print(f"\n[done] best test loss = {best_test_loss:.4f}  out_dir={out_dir}")


if __name__ == "__main__":
    main()
