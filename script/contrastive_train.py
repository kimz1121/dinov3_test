"""DINOv3 patch token 위에 학습하는 task / nuisance disentanglement contrastive 모델.

입력: save_dinov3_patch_repr.py 가 만든 {Task}.hdf5 (clip 당 (n, H, W, D)).
모델: 단일 attention pool → 두 projection head (z_task, z_nuisance), shared trunk.
손실: SupCon(z_task, task_label) + λ_n SupCon(z_nuis, camera_label) + λ_o Orth(z_t, z_n).
분할: task 별 episode 정렬해서 앞 80% train / 뒤 20% test.

사용 예:
    # 기본 (RTX 4070 Ti 가정)
    python script/contrastive_train.py

    # 빠른 sanity check
    python script/contrastive_train.py --epochs 5 --log-every 5
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class SampleIndex:
    """HDF5 위치 + label 모음 — Dataset 인덱싱 핵심."""

    h5_path: str
    task_id: int
    camera: str
    camera_id: int
    ep_idx: int
    demo_key: str
    clip_id: int
    clip_key: str


def discover_h5(h5_root: Path, tasks: list[str] | None) -> list[Path]:
    files = sorted(h5_root.glob("*.hdf5"))
    if tasks:
        files = [p for p in files if p.stem in set(tasks)]
    return files


def build_indices(
    h5_paths: list[Path],
    train_ratio: float,
    cameras_filter: list[str] | None,
) -> tuple[list[SampleIndex], list[SampleIndex], dict[str, int], list[str]]:
    """모든 HDF5 파일 훑어 train/test 인덱스 리스트와 task→id 매핑 생성.

    split: task 별 episode 정렬해서 앞 floor(ratio·L) 를 train.
    Returns:
        train_idx, test_idx, task_to_id, cameras
    """
    tasks_sorted = sorted(p.stem for p in h5_paths)
    task_to_id = {t: i for i, t in enumerate(tasks_sorted)}

    cameras_all: list[str] | None = None
    train: list[SampleIndex] = []
    test: list[SampleIndex] = []

    for h5_path in h5_paths:
        task = h5_path.stem
        task_id = task_to_id[task]
        with h5py.File(h5_path, "r") as f:
            cams_in_file = [c for c in f.attrs["cameras"]]
            cams = cams_in_file if cameras_filter is None else [
                c for c in cams_in_file if c in set(cameras_filter)
            ]
            if cameras_all is None:
                cameras_all = cams
            else:
                if cams != cameras_all:
                    raise SystemExit(
                        f"camera 집합 불일치: {h5_path} 의 {cams} vs 누적 {cameras_all}"
                    )

            demos = sorted(f["data"].keys())
            n_total = len(demos)
            n_train = int(math.floor(train_ratio * n_total))
            train_demos = set(demos[:n_train])

            for demo_key in demos:
                ep_idx = int(demo_key.split("_")[1])
                clip_keys = sorted(f[f"data/{demo_key}"].keys())
                target = train if demo_key in train_demos else test
                for clip_key in clip_keys:
                    clip_id = int(clip_key.split("_")[1])
                    for cam_id, cam in enumerate(cams):
                        target.append(
                            SampleIndex(
                                h5_path=str(h5_path),
                                task_id=task_id,
                                camera=cam,
                                camera_id=cam_id,
                                ep_idx=ep_idx,
                                demo_key=demo_key,
                                clip_id=clip_id,
                                clip_key=clip_key,
                            )
                        )

    if cameras_all is None:
        raise SystemExit("HDF5 가 비었거나 카메라 정보를 못 읽음")

    print(
        f"[INFO] tasks={tasks_sorted}  cameras={cameras_all}  "
        f"train={len(train)} test={len(test)}"
    )
    return train, test, task_to_id, cameras_all


class PatchClipDataset(Dataset):
    """HDF5 lazy-load. clip 의 (n, H, W, D) 를 (n·H·W, D) 토큰 시퀀스로 반환."""

    def __init__(self, indices: list[SampleIndex]):
        self.indices = indices
        # h5py.File 은 worker 별로 lazy open
        self._h5_cache: dict[str, h5py.File] = {}

    def _h5(self, path: str) -> h5py.File:
        if path not in self._h5_cache:
            self._h5_cache[path] = h5py.File(path, "r", swmr=True)
        return self._h5_cache[path]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        s = self.indices[i]
        f = self._h5(s.h5_path)
        arr = f[f"data/{s.demo_key}/{s.clip_key}/{s.camera}"][...]
        # arr: (n, H, W, D)
        n, H, W, D = arr.shape
        patches = torch.from_numpy(arr).float().reshape(n * H * W, D)
        return (
            patches,
            s.task_id,
            s.camera_id,
            s.ep_idx,
            s.clip_id,
        )


# ---------------------------------------------------------------------------
# Balanced sampler — SupCon 은 batch 안에 같은 label 이 2 개 이상 필요
# ---------------------------------------------------------------------------


class BalancedBatchSampler(torch.utils.data.Sampler[list[int]]):
    """(task, camera) cell 별로 균등하게 뽑는 배치 sampler.

    batch_size 가 cell 개수의 배수가 아니어도 동작하지만, 깨끗하게 나뉘는 게
    SupCon positive 수를 균일하게 유지함.
    """

    def __init__(
        self,
        indices: list[SampleIndex],
        batch_size: int,
        num_tasks: int,
        num_cameras: int,
        rng: np.random.Generator,
    ):
        self.batch_size = batch_size
        self.rng = rng

        # cell index 그룹
        self.cells: dict[tuple[int, int], list[int]] = {}
        for i, s in enumerate(indices):
            self.cells.setdefault((s.task_id, s.camera_id), []).append(i)

        n_cells = num_tasks * num_cameras
        if batch_size % n_cells != 0:
            print(
                f"[warn] batch_size {batch_size} not multiple of {n_cells} cells; "
                f"some cells contribute 1 more sample than others"
            )
        self.per_cell = max(1, batch_size // n_cells)
        self.num_tasks = num_tasks
        self.num_cameras = num_cameras

        # epoch 길이: 가장 큰 cell 기준
        max_cell = max(len(v) for v in self.cells.values()) if self.cells else 0
        self.steps_per_epoch = max(1, max_cell // self.per_cell)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        # 각 epoch 마다 cell 별 shuffle, round-robin 으로 per_cell 개씩 뽑음
        cell_perms: dict[tuple[int, int], np.ndarray] = {}
        for key, idxs in self.cells.items():
            arr = np.array(idxs)
            self.rng.shuffle(arr)
            cell_perms[key] = arr

        offsets = {k: 0 for k in self.cells}

        for _ in range(self.steps_per_epoch):
            batch: list[int] = []
            for key, arr in cell_perms.items():
                start = offsets[key]
                end = start + self.per_cell
                if end > len(arr):
                    # 부족하면 wrap around 후 다시 shuffle
                    self.rng.shuffle(arr)
                    offsets[key] = 0
                    start, end = 0, self.per_cell
                batch.extend(arr[start:end].tolist())
                offsets[key] = end
            # batch 크기가 batch_size 와 다르면 남은 자리는 랜덤 cell 에서 채움
            while len(batch) < self.batch_size:
                key = list(self.cells.keys())[
                    self.rng.integers(0, len(self.cells))
                ]
                arr = cell_perms[key]
                pick = arr[self.rng.integers(0, len(arr))]
                batch.append(int(pick))
            self.rng.shuffle(batch)
            yield batch


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class AttentionPool(nn.Module):
    """K 개 learnable query 로 patch token 에 cross-attention 해서 (B, K, D) 반환."""

    def __init__(self, d_model: int = 384, num_queries: int = 4, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.num_queries = num_queries
        self.d_model = d_model

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # patches: (B, N, D)
        B = patches.shape[0]
        kv = self.norm_kv(patches)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)        # (B, K, D)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        out = self.norm_out(out + q)
        return out                                              # (B, K, D)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


class DisentangleModel(nn.Module):
    def __init__(
        self,
        d_model: int = 384,
        num_queries: int = 4,
        d_task: int = 128,
        d_nuis: int = 64,
        hidden: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pool = AttentionPool(d_model, num_queries, num_heads, dropout)
        in_dim = num_queries * d_model
        self.head_task = ProjectionHead(in_dim, hidden, d_task)
        self.head_nuis = ProjectionHead(in_dim, hidden, d_nuis)
        self.config = dict(
            d_model=d_model,
            num_queries=num_queries,
            d_task=d_task,
            d_nuis=d_nuis,
            hidden=hidden,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.pool(patches).flatten(1)
        return self.head_task(h), self.head_nuis(h)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla 2020) — z 는 L2 정규화 가정.

    같은 label 끼리 가깝게, 다른 label 끼리 멀게.
    """
    B = z.shape[0]
    device = z.device
    sim = z @ z.T / temperature                                # (B, B)
    # self-similarity 제거
    eye = torch.eye(B, dtype=torch.bool, device=device)
    sim.masked_fill_(eye, float("-inf"))

    labels = labels.view(-1)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))    # (B, B)
    pos_mask = pos_mask & ~eye

    # log softmax over all non-self — 대각의 -inf 가 곱셈으로 NaN 을 만들지 않도록
    # 마스크 위치를 0 으로 치환
    log_prob = sim - torch.logsumexp(sim, dim=-1, keepdim=True)
    log_prob = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))

    # positive 가 0 개인 row 는 loss 기여 0 (분모 clamp)
    pos_counts = pos_mask.sum(-1).clamp(min=1).float()
    loss_per_row = -log_prob.sum(-1) / pos_counts

    # positive 가 진짜로 0 인 row 는 마스크 아웃
    has_pos = pos_mask.any(-1)
    if not has_pos.any():
        return torch.zeros((), device=device)
    return loss_per_row[has_pos].mean()


def ortho_loss(z_task: torch.Tensor, z_nuis: torch.Tensor) -> torch.Tensor:
    """centered cross-covariance Frobenius² — 두 subspace 가 직교일수록 작아짐."""
    zt = z_task - z_task.mean(0, keepdim=True)
    zn = z_nuis - z_nuis.mean(0, keepdim=True)
    B = zt.shape[0]
    C = zt.T @ zn / max(B - 1, 1)                              # (d_task, d_nuis)
    return (C ** 2).sum()


# ---------------------------------------------------------------------------
# k-NN probe — disentanglement 평가
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_embeddings(
    model: "DisentangleModel",
    loader: DataLoader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """전체 loader 를 돌려 (z_task, z_nuis, task_id, camera_id) 모음 반환."""
    model.eval()
    zs_t, zs_n, ys_t, ys_c = [], [], [], []
    for batch in loader:
        patches, task_id, cam_id, _, _ = batch
        patches = patches.to(device, non_blocking=True)
        z_task, z_nuis = model(patches)
        zs_t.append(z_task.cpu())
        zs_n.append(z_nuis.cpu())
        ys_t.append(task_id)
        ys_c.append(cam_id)
    return torch.cat(zs_t), torch.cat(zs_n), torch.cat(ys_t), torch.cat(ys_c)


def knn_accuracy(
    feat_train: torch.Tensor,
    y_train: torch.Tensor,
    feat_test: torch.Tensor,
    y_test: torch.Tensor,
    k: int = 10,
) -> float:
    """L2-정규화 가정. cosine sim top-k majority vote 정확도."""
    k = min(k, feat_train.shape[0])
    sim = feat_test @ feat_train.T                             # (Nte, Ntr)
    _, idx = sim.topk(k, dim=-1)
    nn_labels = y_train[idx]                                   # (Nte, k)
    pred = nn_labels.mode(dim=-1).values
    return (pred == y_test).float().mean().item()


def run_probes(
    model: "DisentangleModel",
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    k: int = 10,
) -> dict[str, float]:
    """4 종 probe — 좋은 disentangle 의 신호:
        task_on_z_task ↑,  cam_on_z_nuis ↑  (정상 신호 잘 잡힘)
        cam_on_z_task  ↓,  task_on_z_nuis ↓ (반대 신호 누설 적음)
    """
    zt_tr, zn_tr, yt_tr, yc_tr = compute_embeddings(model, train_loader, device)
    zt_te, zn_te, yt_te, yc_te = compute_embeddings(model, test_loader, device)
    return {
        "task_on_z_task": knn_accuracy(zt_tr, yt_tr, zt_te, yt_te, k),
        "cam_on_z_task":  knn_accuracy(zt_tr, yc_tr, zt_te, yc_te, k),
        "task_on_z_nuis": knn_accuracy(zn_tr, yt_tr, zn_te, yt_te, k),
        "cam_on_z_nuis":  knn_accuracy(zn_tr, yc_tr, zn_te, yc_te, k),
    }


# ---------------------------------------------------------------------------
# Train / eval epoch
# ---------------------------------------------------------------------------


def run_epoch(
    model: DisentangleModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    *,
    temperature: float,
    lambda_nuis: float,
    lambda_ortho: float,
    log_every: int = 0,
    epoch: int = 0,
    log_fp=None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    sums = {"loss": 0.0, "L_task": 0.0, "L_nuis": 0.0, "L_orth": 0.0}
    n_steps = 0

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for step, batch in enumerate(loader):
            patches, task_id, cam_id, _, _ = batch
            patches = patches.to(device, non_blocking=True)
            task_id = task_id.to(device)
            cam_id = cam_id.to(device)

            z_task, z_nuis = model(patches)

            L_task = supcon_loss(z_task, task_id, temperature)
            L_nuis = supcon_loss(z_nuis, cam_id, temperature)
            L_orth = ortho_loss(z_task, z_nuis)
            loss = L_task + lambda_nuis * L_nuis + lambda_ortho * L_orth

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            sums["loss"] += loss.item()
            sums["L_task"] += L_task.item()
            sums["L_nuis"] += L_nuis.item()
            sums["L_orth"] += L_orth.item()
            n_steps += 1

            if is_train and log_every and step % log_every == 0:
                msg = {
                    "epoch": epoch,
                    "step": step,
                    "phase": "train",
                    "loss": loss.item(),
                    "L_task": L_task.item(),
                    "L_nuis": L_nuis.item(),
                    "L_orth": L_orth.item(),
                }
                print(
                    f"  ep{epoch:02d} step{step:03d}  "
                    f"loss={loss.item():.4f}  "
                    f"task={L_task.item():.4f}  "
                    f"nuis={L_nuis.item():.4f}  "
                    f"orth={L_orth.item():.4f}"
                )
                if log_fp is not None:
                    log_fp.write(json.dumps(msg) + "\n")
                    log_fp.flush()

    return {k: v / max(n_steps, 1) for k, v in sums.items()}


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------


def make_scheduler(
    optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int
) -> torch.optim.lr_scheduler.LRScheduler:
    """5 epoch warmup + cosine annealing."""

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--patch-h5-root", type=Path, default=Path("data/patch_embeddings"))
    p.add_argument("--tasks", nargs="+", default=None)
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
        help="기본: runs/{timestamp}",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument(
        "--probe-every",
        type=int,
        default=1,
        help="N epoch 마다 k-NN probe 실행 (0 이면 비활성)",
    )
    p.add_argument("--probe-k", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("runs") / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] out_dir={out_dir}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 데이터 ----
    h5_paths = discover_h5(args.patch_h5_root, args.tasks)
    if not h5_paths:
        raise SystemExit(f"no *.hdf5 under {args.patch_h5_root}")
    train_idx, test_idx, task_to_id, cameras = build_indices(
        h5_paths, args.train_ratio, args.cameras
    )

    num_tasks = len(task_to_id)
    num_cameras = len(cameras)

    train_ds = PatchClipDataset(train_idx)
    test_ds = PatchClipDataset(test_idx)

    sampler_rng = np.random.default_rng(args.seed)
    train_sampler = BalancedBatchSampler(
        train_idx, args.batch_size, num_tasks, num_cameras, sampler_rng
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
    # probe 용 — train 전체를 순차로 한 번 인코딩
    probe_train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # ---- 모델 ----
    # D 는 HDF5 attrs 에서 추론
    with h5py.File(h5_paths[0], "r") as f:
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

    # ---- config 저장 (Path 는 str 로 직렬화) ----
    args_dict = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
    }
    config = {
        "args": args_dict,
        "task_to_id": task_to_id,
        "cameras": cameras,
        "h5_paths": [str(p) for p in h5_paths],
        "d_model": d_model,
        "model_config": model.config,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

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
                f"  [probe] task@z_t={probe_stats['task_on_z_task']:.3f}  "
                f"cam@z_t={probe_stats['cam_on_z_task']:.3f}  "
                f"(↓ disentangle)  |  "
                f"cam@z_n={probe_stats['cam_on_z_nuis']:.3f}  "
                f"task@z_n={probe_stats['task_on_z_nuis']:.3f}  (↓ disentangle)"
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
