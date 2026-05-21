"""Quick: fusion no_text/with_text per-class recall."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from contrastive_train import PatchClipDataset, knn_accuracy
from contrastive_train_cross_task import build_cross_task_indices
from contrastive_train_cross_task_fusion import (
    FusionDisentangleModel,
    compute_embeddings_fusion,
)


def main():
    run_dir = Path("/home/iw/dinov3_test/runs/exp2_cross_task_fusion_drop05")
    config = json.loads((run_dir / "config.json").read_text())
    device = "cuda"
    n_cls = int(config["num_subtasks"])
    unified_mapping = config["unified_mapping"]
    h5_paths = [Path(p) for p in config["h5_paths"]]
    seed = int(config["args"]["seed"])
    split_rng = np.random.default_rng(seed)
    train_idx, test_idx, _, _ = build_cross_task_indices(
        h5_paths, Path(config["args"]["patch_h5_root"]), unified_mapping,
        float(config["args"]["train_ratio"]), config["args"].get("cameras"),
        split_rng,
    )
    mc = config["model_config"]
    model = FusionDisentangleModel(**mc).to(device)
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    text_protos = torch.load(run_dir / "text_protos.pt", map_location="cpu")

    bs = int(config["args"]["batch_size"])
    nw = int(config["args"].get("num_workers", 4))
    train_loader = DataLoader(PatchClipDataset(train_idx), batch_size=bs,
                               shuffle=False, num_workers=nw)
    test_loader = DataLoader(PatchClipDataset(test_idx), batch_size=bs,
                              shuffle=False, num_workers=nw)

    for mode in ["with_text", "no_text"]:
        zt_tr, _, yt_tr, _ = compute_embeddings_fusion(
            model, text_protos, train_loader, device, text_mode=mode
        )
        zt_te, _, yt_te, _ = compute_embeddings_fusion(
            model, text_protos, test_loader, device, text_mode=mode
        )
        # knn predict per sample
        Zn_tr = F.normalize(zt_tr, dim=1)
        Zn_te = F.normalize(zt_te, dim=1)
        sim = Zn_te @ Zn_tr.T
        top = sim.topk(k=10, dim=1).indices
        nn_lbl = yt_tr[top]
        preds = []
        for row in nn_lbl:
            vals, counts = torch.unique(row, return_counts=True)
            preds.append(int(vals[counts.argmax()]))
        preds = np.array(preds)
        ytrue = yt_te.cpu().numpy()
        acc = (preds == ytrue).mean()
        rec = np.zeros(n_cls)
        for c in range(n_cls):
            m = ytrue == c
            if m.sum() > 0:
                rec[c] = (preds[m] == c).mean()
        print(f"\n=== mode={mode} knn={acc:.4f} ===")
        for c in range(n_cls):
            print(f"  {c:>2}  recall={rec[c]:.3f}")


if __name__ == "__main__":
    main()
