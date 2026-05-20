"""3.2 Patch-level probe diagnostic.

Visual-only baseline (AttentionPool + SupCon) 의 ceiling 이 0.767 인 이유가:
(A) DINOv3 patch 자체에 정보가 없어서인지, 아니면
(B) AttentionPool / SupCon 이 정보를 추출하지 못한 것인지

이를 가르기 위해 frozen patch 위에 단순 classifier 들로 fully-supervised CE 학습:
- linear_mean      : 시공간 mean → 384 → Linear(384, 13)
- linear_spatial   : 시간 mean (14,14,384) → flatten 75264 → Linear
- linear_delta     : last frame - first frame → mean spatial → 384 → Linear
- mlp_mean         : mean → MLP(384→256→13)
- mlp_spatial      : spatial flatten → MLP(75264→512→13)

각 probe 에서 hot pair (0,1) (2,3) (8,9) recall 을 baseline 0.767 와 비교.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from contrastive_train_cross_task import build_cross_task_indices


class ClipPatchFeats(Dataset):
    """Cross-task 인덱스로부터 5가지 feature variant 를 in-memory 로 캐시."""

    def __init__(self, indices, mode: str):
        self.indices = indices
        self.mode = mode
        self._h5: dict[str, h5py.File] = {}

    def _f(self, path: str) -> h5py.File:
        if path not in self._h5:
            self._h5[path] = h5py.File(path, "r", swmr=True)
        return self._h5[path]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        s = self.indices[i]
        arr = self._f(s.h5_path)[f"data/{s.demo_key}/{s.clip_key}/{s.camera}"][...]
        # arr: (n, H, W, D)
        x = torch.from_numpy(arr).float()  # (n,H,W,D)
        n, H, W, D = x.shape
        if self.mode == "mean":
            feat = x.mean(dim=(0, 1, 2))  # (D,)
        elif self.mode == "spatial":
            feat = x.mean(dim=0).reshape(-1)  # (H*W*D,)
        elif self.mode == "delta":
            feat = (x[-1] - x[0]).mean(dim=(0, 1))  # (D,)
        elif self.mode == "delta_spatial":
            feat = (x[-1] - x[0]).reshape(-1)
        else:
            raise ValueError(self.mode)
        return feat, s.task_id, s.camera_id


def cache_features(loader, n: int, dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X = torch.empty(n, dim, dtype=torch.float32)
    Y = torch.empty(n, dtype=torch.long)
    C = torch.empty(n, dtype=torch.long)
    cur = 0
    for feat, task_id, cam_id in loader:
        b = feat.size(0)
        X[cur : cur + b] = feat
        Y[cur : cur + b] = task_id
        C[cur : cur + b] = cam_id
        cur += b
    return X, Y, C


class LinearProbe(nn.Module):
    def __init__(self, d_in: int, n_cls: int):
        super().__init__()
        self.fc = nn.Linear(d_in, n_cls)

    def forward(self, x):
        return self.fc(x)


class MLPProbe(nn.Module):
    def __init__(self, d_in: int, d_hid: int, n_cls: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_hid, n_cls),
        )

    def forward(self, x):
        return self.net(x)


def train_probe(model, X_tr, Y_tr, X_te, Y_te, *, epochs=60, bs=256, lr=3e-3, wd=1e-4,
                device="cuda") -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n_tr = X_tr.size(0)
    best = 0.0
    best_preds = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr)
        total = 0.0
        for i in range(0, n_tr, bs):
            b = perm[i : i + bs]
            x = X_tr[b].to(device, non_blocking=True)
            y = Y_tr[b].to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        sch.step()

        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, X_te.size(0), 1024):
                xb = X_te[i : i + 1024].to(device)
                preds.append(model(xb).argmax(1).cpu())
            preds = torch.cat(preds)
        acc = (preds == Y_te).float().mean().item()
        if acc > best:
            best = acc
            best_preds = preds.numpy()
    return best, best_preds, X_tr.shape[1] * 1, Y_te.numpy()


def per_class_recall(y_true, y_pred, n_cls):
    rec = np.zeros(n_cls)
    for c in range(n_cls):
        m = y_true == c
        if m.sum() > 0:
            rec[c] = (y_pred[m] == c).mean()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path,
                    default=Path("/home/iw/dinov3_test/runs/exp2_cross_task_visual"))
    ap.add_argument("--patch-h5-root", type=Path,
                    default=Path("/home/iw/dinov3_test/data/patch_embeddings"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    config = json.loads((args.run_dir / "config.json").read_text())
    subtask_labels = {int(k): v for k, v in config["subtask_id_to_label"].items()}
    n_cls = int(config["num_subtasks"])
    cams = config["cameras"]
    unified_mapping = config["unified_mapping"]
    h5_paths = [Path(p) if Path(p).is_absolute() else (args.patch_h5_root.parent.parent / p)
                for p in config["h5_paths"]]
    # adjust to actual filesystem
    h5_paths = [args.patch_h5_root / Path(p).name for p in config["h5_paths"]]
    seed = int(config["args"]["seed"])
    rng = np.random.default_rng(seed)
    train_idx, test_idx, _, _ = build_cross_task_indices(
        h5_paths, args.patch_h5_root, unified_mapping,
        float(config["args"]["train_ratio"]),
        config["args"].get("cameras"), rng,
    )
    print(f"[INFO] train={len(train_idx)}  test={len(test_idx)}  n_cls={n_cls}")

    out_dir = args.run_dir / "diagnostic_probe"
    out_dir.mkdir(exist_ok=True)
    device = args.device

    modes = {
        "mean":          {"dim": 384,       "probes": [("linear", None), ("mlp", 256)]},
        "delta":         {"dim": 384,       "probes": [("linear", None), ("mlp", 256)]},
        "spatial":       {"dim": 14*14*384, "probes": [("linear", None), ("mlp", 512)]},
    }

    results = {}
    HOT_PAIRS = [(0, 1), (2, 3), (8, 9)]

    for mode_name, info in modes.items():
        print(f"\n=== feature mode = {mode_name} (dim={info['dim']}) ===")
        ds_tr = ClipPatchFeats(train_idx, mode_name)
        ds_te = ClipPatchFeats(test_idx, mode_name)
        ld_tr = DataLoader(ds_tr, batch_size=128, shuffle=False, num_workers=4)
        ld_te = DataLoader(ds_te, batch_size=128, shuffle=False, num_workers=4)
        X_tr, Y_tr, _ = cache_features(ld_tr, len(ds_tr), info["dim"])
        X_te, Y_te, _ = cache_features(ld_te, len(ds_te), info["dim"])
        print(f"  cached  X_tr={tuple(X_tr.shape)}  X_te={tuple(X_te.shape)}")

        # normalize per-feature (standardize) for stability on large dim
        mu = X_tr.mean(0, keepdim=True)
        sd = X_tr.std(0, keepdim=True).clamp(min=1e-6)
        X_tr = (X_tr - mu) / sd
        X_te = (X_te - mu) / sd

        for probe_name, hid in info["probes"]:
            if probe_name == "linear":
                model = LinearProbe(info["dim"], n_cls)
            else:
                model = MLPProbe(info["dim"], hid, n_cls)

            n_params = sum(p.numel() for p in model.parameters())
            print(f"  --> {probe_name} (params={n_params:,})")
            acc, preds, _, ytrue = train_probe(
                model, X_tr, Y_tr, X_te, Y_te,
                epochs=args.epochs, bs=args.bs, device=device,
            )
            rec = per_class_recall(ytrue, preds, n_cls)
            hot_rec = {f"{a}_{b}": (float(rec[a]), float(rec[b])) for a, b in HOT_PAIRS}
            print(f"     overall acc={acc:.3f}")
            print(f"     hot recall  (0,1)={rec[0]:.2f},{rec[1]:.2f}   "
                  f"(2,3)={rec[2]:.2f},{rec[3]:.2f}   "
                  f"(8,9)={rec[8]:.2f},{rec[9]:.2f}")
            key = f"{mode_name}_{probe_name}"
            results[key] = {
                "acc": float(acc),
                "params": int(n_params),
                "per_class_recall": rec.tolist(),
                "hot_pair_recall": hot_rec,
            }

    # also: a tiny non-spatial pseudo-classifier: nearest-mean (no training) for grounding
    print("\n=== nearest-class-mean (no training, mean feature) ===")
    ds_tr = ClipPatchFeats(train_idx, "mean")
    ds_te = ClipPatchFeats(test_idx, "mean")
    ld_tr = DataLoader(ds_tr, batch_size=128, shuffle=False, num_workers=4)
    ld_te = DataLoader(ds_te, batch_size=128, shuffle=False, num_workers=4)
    X_tr, Y_tr, _ = cache_features(ld_tr, len(ds_tr), 384)
    X_te, Y_te, _ = cache_features(ld_te, len(ds_te), 384)
    Xn = F.normalize(X_tr, dim=1)
    centroids = torch.zeros(n_cls, 384)
    for c in range(n_cls):
        m = Y_tr == c
        if m.sum() > 0:
            centroids[c] = F.normalize(Xn[m].mean(0), dim=0)
    sims = F.normalize(X_te, dim=1) @ centroids.T
    preds = sims.argmax(1).numpy()
    ytrue = Y_te.numpy()
    rec = per_class_recall(ytrue, preds, n_cls)
    acc = (preds == ytrue).mean()
    print(f"  NCM acc={acc:.3f}  (0,1)={rec[0]:.2f},{rec[1]:.2f}  "
          f"(2,3)={rec[2]:.2f},{rec[3]:.2f}  (8,9)={rec[8]:.2f},{rec[9]:.2f}")
    results["mean_ncm"] = {"acc": float(acc), "per_class_recall": rec.tolist()}

    results["baseline_visual_only_knn"] = 0.7673
    out_json = out_dir / "probe_summary.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[done] -> {out_json}")

    print("\n" + "=" * 78)
    print("SUMMARY (vs visual-only baseline knn = 0.767)")
    print("=" * 78)
    print(f"{'probe':<28}  {'acc':>6}  {'(0,1)':>10}  {'(2,3)':>10}  {'(8,9)':>10}")
    for k, v in results.items():
        if k in ("baseline_visual_only_knn",):
            continue
        rec = v["per_class_recall"]
        print(f"{k:<28}  {v['acc']:>6.3f}  "
              f"{rec[0]:.2f}/{rec[1]:.2f}    {rec[2]:.2f}/{rec[3]:.2f}    {rec[8]:.2f}/{rec[9]:.2f}")


if __name__ == "__main__":
    main()
