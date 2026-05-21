"""Per-class within−cross cosine across 4 settings (visual, dual VL, fusion no_text/with_text).

Each setting:
  - compute test-set z_task, L2-norm
  - per class c:
      within[c] = mean cos(i,j) for i!=j with y_i=y_j=c
      cross[c]  = mean cos(i,j) for y_i=c, y_j!=c
  - write per_class_winxcross.json into each run dir's analysis dir
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from contrastive_train import PatchClipDataset, DisentangleModel
from contrastive_train_cross_task import build_cross_task_indices
from contrastive_train_cross_task_fusion import (
    FusionDisentangleModel,
    compute_embeddings_fusion,
)
from contrastive_train_subtask_vl import VLModel


N_CLS = 13


def per_class_within_cross(Z: torch.Tensor, y: torch.Tensor, n_cls: int):
    """Z: (N, d) L2-norm.  y: (N,)  → (within[c], cross[c]) for c in range(n_cls)."""
    Z = F.normalize(Z, dim=1)
    sim = Z @ Z.T          # (N, N)
    y_np = y.cpu().numpy()
    within = np.zeros(n_cls)
    cross = np.zeros(n_cls)
    for c in range(n_cls):
        mask_c = y_np == c
        if mask_c.sum() < 2:
            continue
        idx_c = np.where(mask_c)[0]
        idx_o = np.where(~mask_c)[0]
        # within: upper triangle within class c
        S_cc = sim[np.ix_(idx_c, idx_c)].cpu().numpy()
        iu = np.triu_indices(len(idx_c), k=1)
        within[c] = float(S_cc[iu].mean())
        # cross: all pairs (c, other)
        S_co = sim[np.ix_(idx_c, idx_o)].cpu().numpy()
        cross[c] = float(S_co.mean())
    return within, cross


def build_split(config):
    h5_paths = [Path(p) for p in config["h5_paths"]]
    seed = int(config["args"]["seed"])
    rng = np.random.default_rng(seed)
    unified = config["unified_mapping"]
    train_idx, test_idx, _, _ = build_cross_task_indices(
        h5_paths,
        Path(config["args"]["patch_h5_root"]),
        unified,
        float(config["args"]["train_ratio"]),
        config["args"].get("cameras"),
        rng,
    )
    return train_idx, test_idx


def setting_visual(device="cuda"):
    run = Path("/home/iw/dinov3_test/runs/exp2_cross_task_visual")
    cfg = json.loads((run / "config.json").read_text())
    mc = cfg["model_config"]
    model = DisentangleModel(**mc).to(device).eval()
    ckpt = torch.load(run / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    _, test_idx = build_split(cfg)
    bs = int(cfg["args"]["batch_size"])
    loader = DataLoader(PatchClipDataset(test_idx), batch_size=bs,
                        shuffle=False, num_workers=4)
    Z, Y = [], []
    with torch.no_grad():
        for batch in loader:
            patches, task_id, _, _, _ = batch
            patches = patches.to(device, non_blocking=True)
            zt, _ = model(patches)
            Z.append(F.normalize(zt, dim=1).cpu())
            Y.append(task_id)
    return torch.cat(Z), torch.cat(Y), run


def setting_dual_vl(device="cuda"):
    run = Path("/home/iw/dinov3_test/runs/exp2_cross_task_vl_alignment_infonce")
    cfg = json.loads((run / "config.json").read_text())
    mc = cfg["model_config"]
    visual = DisentangleModel(**mc)
    model = VLModel(visual, text_dim=int(cfg["text_dim"]),
                    d_task=int(mc["d_task"]),
                    hidden=int(cfg["args"]["text_proj_hidden"])).to(device).eval()
    ckpt = torch.load(run / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["vl_model"])
    _, test_idx = build_split(cfg)
    bs = int(cfg["args"]["batch_size"])
    loader = DataLoader(PatchClipDataset(test_idx), batch_size=bs,
                        shuffle=False, num_workers=4)
    Z, Y = [], []
    with torch.no_grad():
        for batch in loader:
            patches, task_id, _, _, _ = batch
            patches = patches.to(device, non_blocking=True)
            zt, _ = model(patches)
            Z.append(F.normalize(zt, dim=1).cpu())
            Y.append(task_id)
    return torch.cat(Z), torch.cat(Y), run


def setting_fusion(mode: str, device="cuda"):
    """mode = 'no_text' or 'with_text'"""
    run = Path("/home/iw/dinov3_test/runs/exp2_cross_task_fusion_drop05")
    cfg = json.loads((run / "config.json").read_text())
    mc = cfg["model_config"]
    model = FusionDisentangleModel(**mc).to(device).eval()
    ckpt = torch.load(run / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    text_protos = torch.load(run / "text_protos.pt", map_location="cpu")
    _, test_idx = build_split(cfg)
    bs = int(cfg["args"]["batch_size"])
    loader = DataLoader(PatchClipDataset(test_idx), batch_size=bs,
                        shuffle=False, num_workers=4)
    zt, _, yt, _ = compute_embeddings_fusion(model, text_protos, loader, device,
                                              text_mode=mode)
    Z = F.normalize(zt, dim=1).cpu()
    return Z, yt.cpu(), run


def main():
    device = "cuda"
    out = {}

    for name, fn in [
        ("visual",       lambda: setting_visual(device)),
        ("dual_vl",      lambda: setting_dual_vl(device)),
        ("fusion_no_text",   lambda: setting_fusion("no_text", device)),
        ("fusion_with_text", lambda: setting_fusion("with_text", device)),
    ]:
        print(f"\n=== {name} ===")
        Z, Y, run = fn()
        within, cross = per_class_within_cross(Z, Y, N_CLS)
        diff = within - cross
        out[name] = {
            "within": [float(x) for x in within],
            "cross":  [float(x) for x in cross],
            "diff":   [float(x) for x in diff],
        }
        for c in range(N_CLS):
            print(f"  {c:>2}  within={within[c]:.3f}  cross={cross[c]:.3f}  diff={diff[c]:+.3f}")

    save_to = Path("/home/iw/dinov3_test/runs/_per_class_winxcross.json")
    save_to.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] -> {save_to}")


if __name__ == "__main__":
    main()
