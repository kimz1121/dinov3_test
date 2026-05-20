"""Exp 2 cross-task FUSION 분석.

with_text / no_text 두 mode 각각으로 inference 해서 동일한 4가지 plot 생성.
analyze_exp2.py 의 plot 함수들을 그대로 재사용.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_exp1 import (
    collect_z,    # 안 씀 — fusion 은 별도 forward
    knn_predict,
    plot_tsne_grid,
)
from analyze_exp2 import (
    plot_confusion_cross,
    plot_episode_gram_cross,
    plot_sample_gram_cross,
)
from contrastive_train import PatchClipDataset, knn_accuracy
from contrastive_train_cross_task import build_cross_task_indices
from contrastive_train_cross_task_fusion import (
    FusionDisentangleModel,
    compute_embeddings_fusion,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--ckpt", type=str, default="best.pt")
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--device", default=None)
    return p.parse_args()


def analyze_fusion(args: argparse.Namespace) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}  knn_k={args.knn_k}")

    config = json.loads((args.run_dir / "config.json").read_text())
    subtask_to_instr = {int(k): v for k, v in config["subtask_id_to_label"].items()}
    n_cls = int(config["num_subtasks"])
    cams = config["cameras"]
    print(f"[INFO] cameras={cams}")

    mc = config["model_config"]
    model = FusionDisentangleModel(**mc).to(device)
    ckpt = torch.load(args.run_dir / args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    text_protos = torch.load(args.run_dir / "text_protos.pt", map_location="cpu")

    h5_paths = [Path(p) for p in config["h5_paths"]]
    seed = int(config["args"]["seed"])
    split_rng = np.random.default_rng(seed)

    unified_mapping = config["unified_mapping"]
    train_idx, test_idx, _, _ = build_cross_task_indices(
        h5_paths, Path(config["args"]["patch_h5_root"]), unified_mapping,
        float(config["args"]["train_ratio"]), config["args"].get("cameras"),
        split_rng,
    )
    train_ds = PatchClipDataset(train_idx)
    test_ds = PatchClipDataset(test_idx)
    nw = int(config["args"].get("num_workers", 4))
    bs = int(config["args"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=nw,
                              pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw,
                             pin_memory=(device == "cuda"))

    # also need ep_idx
    ep_te = np.asarray([s.ep_idx for s in test_idx])

    out_root = args.run_dir / "analysis_exp2_fusion"
    out_root.mkdir(exist_ok=True)

    summary = {}
    for mode in ["with_text", "no_text"]:
        print(f"\n=== mode={mode} ===")
        zt_tr_t, zn_tr_t, yt_tr_t, yc_tr_t = compute_embeddings_fusion(
            model, text_protos, train_loader, device, text_mode=mode
        )
        zt_te_t, zn_te_t, yt_te_t, yc_te_t = compute_embeddings_fusion(
            model, text_protos, test_loader, device, text_mode=mode
        )

        # to numpy
        Zt_tr = zt_tr_t.cpu().numpy(); Zn_tr = zn_tr_t.cpu().numpy()
        Zt_te = zt_te_t.cpu().numpy(); Zn_te = zn_te_t.cpu().numpy()
        yt_tr = yt_tr_t.cpu().numpy(); yt_te = yt_te_t.cpu().numpy()
        yc_tr = yc_tr_t.cpu().numpy(); yc_te = yc_te_t.cpu().numpy()

        # knn metrics
        knn_acc = knn_accuracy(zt_tr_t, yt_tr_t, zt_te_t, yt_te_t, args.knn_k)
        pred_te = knn_predict(Zt_tr, yt_tr, Zt_te, args.knn_k)
        cm = np.zeros((n_cls, n_cls), dtype=np.int64)
        for t, p in zip(yt_te, pred_te):
            cm[int(t), int(p)] += 1

        out = out_root / mode
        out.mkdir(exist_ok=True)

        # confusion
        labels_list = [subtask_to_instr[i] for i in sorted(subtask_to_instr.keys())]
        plot_confusion_cross(cm, labels_list, args.knn_k, knn_acc,
                              out / "confusion_matrix.png")
        # episode gram + within/cross stats
        ep_id_te = ep_te
        # average z_task per episode for episode-gram
        uniq_eps = np.unique(ep_id_te)
        ep_z = np.stack([Zt_te[ep_id_te == e].mean(0) for e in uniq_eps])
        ep_z /= np.linalg.norm(ep_z, axis=1, keepdims=True).clip(1e-8)
        ep_sid = np.array([yt_te[ep_id_te == e][0] for e in uniq_eps])
        order = np.argsort(ep_sid, kind="stable")
        ep_z_sorted = ep_z[order]
        ep_sid_sorted = ep_sid[order]
        ep_gram = ep_z_sorted @ ep_z_sorted.T
        n_ep = ep_gram.shape[0]
        within_mask = ep_sid_sorted[:, None] == ep_sid_sorted[None, :]
        eye = np.eye(n_ep, dtype=bool)
        within = ep_gram[within_mask & ~eye].mean() if (within_mask & ~eye).any() else 0.0
        cross = ep_gram[~within_mask].mean() if (~within_mask).any() else 0.0
        stats = {"within": float(within), "cross": float(cross)}
        plot_episode_gram_cross(ep_gram, ep_sid_sorted, subtask_to_instr, stats,
                                 out / "episode_gram.png")
        # sample gram
        plot_sample_gram_cross(Zt_te, Zn_te, yt_te, yc_te, ep_id_te,
                                subtask_to_instr, out / "sample_gram.png",
                                camera_names=cams)
        # tsne
        plot_tsne_grid(Zt_te, Zn_te, yt_te, yc_te,
                        subtask_to_instr, cams,
                        f"{config.get('experiment', 'fusion')} [{mode}]",
                        out / "tsne_grid.png")

        delta = within - cross
        print(f"  knn_acc={knn_acc:.3f}  within={within:.3f}  cross={cross:.3f}  Δ={delta:+.3f}")
        summary[mode] = {
            "knn_accuracy": float(knn_acc),
            "within_subtask_cos": float(within),
            "cross_subtask_cos": float(cross),
            "within_minus_cross": float(delta),
        }

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] summary -> {summary_path}")


def main() -> None:
    args = parse_args()
    analyze_fusion(args)


if __name__ == "__main__":
    main()
