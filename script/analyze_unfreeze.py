"""P1 unfreeze 학습 결과 분석.

- kNN(10) overall + per-class recall
- hot pair (0,1)(2,3)(8,9) recall 변화 vs baseline
- attention entropy + pair attention cosine 비교
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

mp.set_sharing_strategy("file_system")

from contrastive_train_unfreeze import (
    RawClipDataset,
    UnfreezeDinoModel,
    compute_embeddings,
    knn_predict,
)
from contrastive_train_cross_task import build_cross_task_indices


def per_class_recall(y_true, y_pred, n_cls):
    rec = np.zeros(n_cls)
    for c in range(n_cls):
        m = y_true == c
        if m.sum() > 0:
            rec[c] = (y_pred[m] == c).mean()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=str, default="best.pt")
    ap.add_argument("--clip-root", type=Path,
                    default=Path("/home/iw/dinov3_test/data/robocasa_clips"))
    ap.add_argument("--patch-h5-root", type=Path,
                    default=Path("/home/iw/dinov3_test/data/patch_embeddings"))
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    config = json.loads((args.run_dir / "config.json").read_text())
    n_cls = int(config["num_subtasks"])
    cams = config["cameras"]
    labels = {int(k): v for k, v in config["subtask_id_to_label"].items()}
    unified_mapping = config["unified_mapping"]
    h5_paths = [args.patch_h5_root / Path(p).name for p in config["h5_paths"]]
    seed = int(config["args"]["seed"])
    rng = np.random.default_rng(seed)
    train_idx, test_idx, _, _ = build_cross_task_indices(
        h5_paths, args.patch_h5_root, unified_mapping,
        float(config["args"]["train_ratio"]),
        config["args"].get("cameras"), rng,
    )
    print(f"[INFO] train={len(train_idx)}  test={len(test_idx)}  n_cls={n_cls}")

    mc = dict(config["model_config"])
    mc.pop("d_model", None)  # inferred from backbone
    model = UnfreezeDinoModel(**mc).to(args.device).eval()
    ckpt = torch.load(args.run_dir / args.ckpt, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"[INFO] loaded ckpt epoch={ckpt.get('epoch', '?')}")

    bs = int(config["args"]["batch_size"])
    nw = 2
    train_loader = DataLoader(RawClipDataset(train_idx, args.clip_root),
                              batch_size=bs, shuffle=False, num_workers=nw,
                              persistent_workers=True)
    test_loader = DataLoader(RawClipDataset(test_idx, args.clip_root),
                             batch_size=bs, shuffle=False, num_workers=nw,
                             persistent_workers=True)

    Zt_tr, Y_tr, _ = compute_embeddings(model, train_loader, args.device, torch.bfloat16)
    Zt_te, Y_te, C_te = compute_embeddings(model, test_loader, args.device, torch.bfloat16)
    Zt_tr_g = Zt_tr.to(args.device)
    Zt_te_g = Zt_te.to(args.device)
    Y_tr_g = Y_tr.to(args.device)
    Y_te_g = Y_te.to(args.device)

    # overall knn
    Zn_tr = F.normalize(Zt_tr_g, dim=1)
    Zn_te = F.normalize(Zt_te_g, dim=1)
    sim = Zn_te @ Zn_tr.T
    top = sim.topk(k=args.knn_k, dim=1).indices
    nn_lbl = Y_tr_g[top]
    preds = []
    for row in nn_lbl:
        vals, counts = torch.unique(row, return_counts=True)
        preds.append(int(vals[counts.argmax()]))
    preds = np.array(preds)
    ytrue = Y_te.numpy()
    acc = (preds == ytrue).mean()
    print(f"\n=== knn(10) accuracy = {acc:.4f}")

    rec = per_class_recall(ytrue, preds, n_cls)
    print("\nper-class recall:")
    for c in range(n_cls):
        marker = "*" if c in {0, 1, 2, 3, 8, 9} else " "
        print(f"  {marker} {c:>2}  recall={rec[c]:.3f}  ({labels[c][:60]})")

    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(ytrue, preds):
        cm[int(t), int(p)] += 1
    row_tot = cm.sum(axis=1)

    # hot pairs symmetric rate
    pairs = []
    for i in range(n_cls):
        for j in range(i + 1, n_cls):
            denom = row_tot[i] + row_tot[j]
            confused = cm[i, j] + cm[j, i]
            score = confused / max(denom, 1)
            pairs.append((score, confused, i, j))
    pairs.sort(reverse=True)
    print("\nTop-8 confused pairs (P1 unfreeze):")
    for s, c, i, j in pairs[:8]:
        print(f"  ({i:>2},{j:>2}) rate={s:.3f} cnt={c}  "
              f"{labels[i][:32]:32s} <-> {labels[j][:32]}")

    # baseline comparison
    baseline_recall = {
        0: 0.312, 1: 0.673, 2: 0.125, 3: 0.733, 4: 0.952, 5: 0.881, 6: 0.798,
        7: 0.913, 8: 0.405, 9: 0.410, 10: 0.901, 11: 0.894, 12: 0.902,
    }
    print("\nHot pair recall  Δ vs baseline (visual-only 0.767):")
    for pair in [(0, 1), (2, 3), (8, 9)]:
        a, b = pair
        ba, bb = baseline_recall[a], baseline_recall[b]
        da, db = rec[a] - ba, rec[b] - bb
        print(f"  ({a},{b})  baseline {ba:.3f}/{bb:.3f}  ->  "
              f"P1 {rec[a]:.3f}/{rec[b]:.3f}  (Δ {da:+.3f}/{db:+.3f})")

    out_dir = args.run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({
        "knn_accuracy": float(acc),
        "per_class_recall": {int(c): float(rec[c]) for c in range(n_cls)},
        "top_pairs": [{"i": int(i), "j": int(j), "rate": float(s), "count": int(c)}
                      for s, c, i, j in pairs[:12]],
        "baseline_knn_acc": 0.7673,
        "baseline_per_class_recall": baseline_recall,
    }, indent=2))
    np.save(out_dir / "confusion_matrix.npy", cm)
    print(f"\n[saved] -> {out_dir}/summary.json")


if __name__ == "__main__":
    main()
