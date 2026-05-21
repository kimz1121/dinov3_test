"""B-1 (unfreeze + VL) 결과 분석 — P1 결과를 baseline 으로 비교."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

mp.set_sharing_strategy("file_system")

from contrastive_train_unfreeze import RawClipDataset
from contrastive_train_unfreeze_vl import UnfreezeVLModel, compute_embeddings_unfreeze
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
    ap.add_argument("--ckpt", type=str, default="ckpt_epoch_19.pt")
    ap.add_argument("--clip-root", type=Path,
                    default=Path("/home/iw/dinov3_test/data/robocasa_clips"))
    ap.add_argument("--patch-h5-root", type=Path,
                    default=Path("/home/iw/dinov3_test/data/patch_embeddings"))
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    config = json.loads((args.run_dir / "config.json").read_text())
    n_cls = int(config["num_subtasks"])
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

    # filter mc keys for UnfreezeVLModel
    mc = dict(config["model_config"])
    text_dim = mc.pop("text_dim")
    text_proj_hidden = mc.pop("text_proj_hidden", 256)
    mc.pop("d_model", None)
    model = UnfreezeVLModel(
        text_dim=text_dim, text_proj_hidden=text_proj_hidden, **mc,
    ).to(args.device).eval()
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

    Zt_tr, Y_tr, _ = compute_embeddings_unfreeze(model, train_loader, args.device, torch.bfloat16)
    Zt_te, Y_te, _ = compute_embeddings_unfreeze(model, test_loader, args.device, torch.bfloat16)

    # knn predict
    Zn_tr = F.normalize(Zt_tr.to(args.device), dim=1)
    Zn_te = F.normalize(Zt_te.to(args.device), dim=1)
    sim = Zn_te @ Zn_tr.T
    top = sim.topk(k=args.knn_k, dim=1).indices
    nn_lbl = Y_tr.to(args.device)[top]
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

    pairs = []
    for i in range(n_cls):
        for j in range(i + 1, n_cls):
            denom = row_tot[i] + row_tot[j]
            confused = cm[i, j] + cm[j, i]
            score = confused / max(denom, 1)
            pairs.append((score, confused, i, j))
    pairs.sort(reverse=True)
    print("\nTop-8 confused pairs (B-1):")
    for s, c, i, j in pairs[:8]:
        print(f"  ({i:>2},{j:>2}) rate={s:.3f} cnt={c}  "
              f"{labels[i][:32]:32s} <-> {labels[j][:32]}")

    # 3-way comparison
    baseline_recall = {
        0: 0.312, 1: 0.673, 2: 0.125, 3: 0.733, 4: 0.952, 5: 0.881, 6: 0.798,
        7: 0.913, 8: 0.405, 9: 0.410, 10: 0.901, 11: 0.894, 12: 0.902,
    }
    p1_recall = {
        0: 0.388, 1: 0.670, 2: 0.226, 3: 0.753, 4: 0.944, 5: 0.958, 6: 0.841,
        7: 0.969, 8: 0.644, 9: 0.597, 10: 0.958, 11: 0.909, 12: 0.926,
    }
    print("\n=== 3-way recall comparison ===")
    print(f"{'id':>3}  {'baseline':>9}  {'P1':>7}  {'B1':>7}  {'P1-bl':>7}  {'B1-P1':>7}  task")
    for c in range(n_cls):
        bl, p1 = baseline_recall[c], p1_recall[c]
        b1 = float(rec[c])
        d1 = p1 - bl
        d2 = b1 - p1
        m = "*" if c in {0,1,2,3,8,9} else " "
        print(f"  {m}{c:>2}  {bl:>9.3f}  {p1:>7.3f}  {b1:>7.3f}  {d1:>+7.3f}  {d2:>+7.3f}  {labels[c][:42]}")

    print("\n=== Hot pair summary ===")
    for pair in [(0, 1), (2, 3), (8, 9)]:
        a, b = pair
        ba, bb = baseline_recall[a], baseline_recall[b]
        pa, pb = p1_recall[a], p1_recall[b]
        b1a, b1b = float(rec[a]), float(rec[b])
        avg_bl = (ba + bb) / 2
        avg_p1 = (pa + pb) / 2
        avg_b1 = (b1a + b1b) / 2
        print(f"  pair ({a},{b}): avg recall  baseline {avg_bl:.3f}  ->  "
              f"P1 {avg_p1:.3f} (Δ {avg_p1-avg_bl:+.3f})  ->  "
              f"B-1 {avg_b1:.3f} (Δ {avg_b1-avg_p1:+.3f})")

    out_dir = args.run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({
        "knn_accuracy": float(acc),
        "per_class_recall": {int(c): float(rec[c]) for c in range(n_cls)},
        "top_pairs": [{"i": int(i), "j": int(j), "rate": float(s), "count": int(c)}
                      for s, c, i, j in pairs[:12]],
        "baseline_knn_acc": 0.7673,
        "baseline_per_class_recall": baseline_recall,
        "p1_per_class_recall": p1_recall,
    }, indent=2))
    np.save(out_dir / "confusion_matrix.npy", cm)
    print(f"\n[saved] -> {out_dir}/summary.json")


if __name__ == "__main__":
    main()
