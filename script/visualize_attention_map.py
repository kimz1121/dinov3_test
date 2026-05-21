"""3.3 AttentionPool attention map 시각화.

각 hot pair / 정상 class 의 test sample 에서 4-query × 4-frame attention 을 추출,
14×14 grid heatmap 으로 raw image 위에 overlay.

추가 진단:
- per-query attention entropy (낮을수록 focused)
- pair 간 attention map cosine 유사도 (높을수록 confuse 사례)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from contrastive_train import DisentangleModel
from contrastive_train_cross_task import build_cross_task_indices


def patch_attention_pool(model: DisentangleModel) -> None:
    """nn.MultiheadAttention 의 forward 를 wrapping 해서 weights 저장."""
    pool = model.pool
    pool._attn_weights = None
    orig = pool.attn

    def forward(q, k, v, *args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = True
        out, w = nn.MultiheadAttention.forward(orig, q, k, v, *args, **kwargs)
        pool._attn_weights = w  # (B, K, N)
        return out, w

    # bind
    import types
    orig.forward = types.MethodType(forward, orig)


def load_clip_pngs(clip_root: Path, task: str, camera: str,
                   ep_idx: int, clip_id: int, n_frames: int = 4) -> list[np.ndarray]:
    dir_ = clip_root / task / camera
    imgs = []
    for f in range(n_frames):
        p = dir_ / f"ep{ep_idx:03d}_clip{clip_id:02d}_f{f:02d}.png"
        imgs.append(np.asarray(Image.open(p).convert("RGB")))
    return imgs


def pick_samples(test_idx, target_subtasks, k_per: int = 3, prefer_camera: str | None = None,
                 seed: int = 0):
    rng = np.random.default_rng(seed)
    picked = {s: [] for s in target_subtasks}
    pool = {s: [] for s in target_subtasks}
    for i, idx in enumerate(test_idx):
        if idx.task_id in target_subtasks:
            if prefer_camera and idx.camera != prefer_camera:
                continue
            pool[idx.task_id].append((i, idx))
    for s in target_subtasks:
        if not pool[s]:
            continue
        choice = rng.choice(len(pool[s]), size=min(k_per, len(pool[s])), replace=False)
        picked[s] = [pool[s][int(c)] for c in choice]
    return picked


def compute_entropy(w: np.ndarray) -> float:
    """w: (N,) normalized prob — Shannon entropy in nats."""
    p = np.clip(w, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path,
                    default=Path("/home/iw/dinov3_test/runs/exp2_cross_task_visual"))
    ap.add_argument("--clip-root", type=Path,
                    default=Path("/home/iw/dinov3_test/data/robocasa_clips"))
    ap.add_argument("--patch-h5-root", type=Path,
                    default=Path("/home/iw/dinov3_test/data/patch_embeddings"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--k-per", type=int, default=3)
    args = ap.parse_args()

    config = json.loads((args.run_dir / "config.json").read_text())
    n_cls = int(config["num_subtasks"])
    cams = config["cameras"]
    labels = {int(k): v for k, v in config["subtask_id_to_label"].items()}
    unified_mapping = config["unified_mapping"]
    h5_paths = [args.patch_h5_root / Path(p).name for p in config["h5_paths"]]
    seed = int(config["args"]["seed"])
    rng = np.random.default_rng(seed)
    train_idx, test_idx, _, task_order = build_cross_task_indices(
        h5_paths, args.patch_h5_root, unified_mapping,
        float(config["args"]["train_ratio"]),
        config["args"].get("cameras"), rng,
    )
    # subtask_id (task_id in SampleIndex) -> top-level task name
    # but SampleIndex.task_id is unified sub-task id (per build_cross_task_indices), not raw task index.
    # We need raw task to find PNG dir. Check the field.
    # From config it stores 'sub_task_to_label' which is unified.
    # build_cross_task_indices likely sets task_id = unified id.

    # Find raw task name per sample via h5_path stem
    sample_task = {i: Path(idx.h5_path).stem for i, idx in enumerate(test_idx)}

    # load model
    mc = config["model_config"]
    model = DisentangleModel(**mc).to(args.device).eval()
    ckpt = torch.load(args.run_dir / "best.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    # monkey-patch attention to expose weights
    import types
    import torch.nn as nn

    pool = model.pool
    orig_attn = pool.attn
    pool._attn_weights = None

    def attn_forward(self, query, key, value, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = True
        out, w = nn.MultiheadAttention.forward(self, query, key, value, **kwargs)
        pool._attn_weights = w
        return out, w

    orig_attn.forward = types.MethodType(attn_forward, orig_attn)

    # sample selection: hot pairs + a confident class (4 CheesyBread)
    target_subtasks = [0, 1, 2, 3, 4, 8, 9]
    picked = pick_samples(test_idx, target_subtasks, k_per=args.k_per,
                          prefer_camera="robot0_agentview_left", seed=0)

    out_dir = args.run_dir / "attention_maps"
    out_dir.mkdir(exist_ok=True)
    K = mc.get("num_queries", 4)
    n_frames = 4
    H = W = 14

    # === per-sample overlay figures ===
    entropy_log = {}
    attn_cache = {s: [] for s in target_subtasks}  # for pair similarity

    for s in target_subtasks:
        for sample_pos, (gi, idx) in enumerate(picked[s]):
            # load patches
            with h5py.File(idx.h5_path, "r") as f:
                arr = f[f"data/{idx.demo_key}/{idx.clip_key}/{idx.camera}"][...]
            patches = torch.from_numpy(arr).float().reshape(1, n_frames * H * W, mc["d_model"]).to(args.device)

            # forward
            with torch.no_grad():
                z_task, _ = model(patches)
            w = pool._attn_weights.cpu().numpy()  # (1, K, N=4*196=784)
            w = w[0]  # (K, 784)
            attn_per_q = w.reshape(K, n_frames, H, W)

            # entropy per query
            ents = [compute_entropy(w[k]) for k in range(K)]
            entropy_log.setdefault(s, []).append(ents)

            # cache (mean over queries) for pair similarity
            attn_cache[s].append(w.mean(0))  # (784,)

            # load images
            task_name = Path(idx.h5_path).stem
            try:
                imgs = load_clip_pngs(args.clip_root, task_name, idx.camera,
                                      idx.ep_idx, idx.clip_id, n_frames=n_frames)
            except FileNotFoundError:
                imgs = None

            fig, axes = plt.subplots(K, n_frames + 1, figsize=(2.0 * (n_frames + 1), 2.0 * K))
            if K == 1:
                axes = axes[None, :]
            for q in range(K):
                axes[q, 0].axis("off")
                axes[q, 0].text(0.5, 0.5, f"Q{q}\nH={ents[q]:.2f}", ha="center", va="center",
                                 fontsize=10)
                for fi in range(n_frames):
                    ax = axes[q, fi + 1]
                    ax.axis("off")
                    if imgs is not None:
                        ax.imshow(imgs[fi])
                    # upsample 14x14 → image size
                    am = attn_per_q[q, fi]
                    am = (am - am.min()) / (am.max() - am.min() + 1e-8)
                    if imgs is not None:
                        # overlay
                        am_up = np.array(Image.fromarray((am * 255).astype(np.uint8))
                                          .resize((imgs[fi].shape[1], imgs[fi].shape[0]),
                                                  Image.BILINEAR)) / 255.0
                        ax.imshow(am_up, cmap="jet", alpha=0.45)
                    else:
                        ax.imshow(am, cmap="jet")
                    if q == 0:
                        ax.set_title(f"f{fi}", fontsize=9)
            short_label = labels[s][:46] + ("…" if len(labels[s]) > 46 else "")
            fig.suptitle(f"subtask {s}: {short_label}\nep{idx.ep_idx}_clip{idx.clip_id}_{idx.camera}",
                          fontsize=10)
            fig.tight_layout()
            out_path = out_dir / f"st{s:02d}_sample{sample_pos}.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

    # === pair similarity ===
    HOT_PAIRS = [(0, 1), (2, 3), (8, 9), (4, 8)]  # last is sanity (different task)
    sim_log = {}
    for a, b in HOT_PAIRS:
        if not attn_cache[a] or not attn_cache[b]:
            continue
        A = np.stack(attn_cache[a])  # (k_a, 784)
        B = np.stack(attn_cache[b])
        # cosine sim between mean attention vectors
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        sim = float((An @ Bn.T).mean())
        sim_log[f"{a}_{b}"] = sim
        print(f"  attention cos (subtask {a} ↔ {b}) = {sim:.3f}")

    # entropy summary
    entropy_means = {int(s): [float(np.mean([e[k] for e in v])) for k in range(K)]
                     for s, v in entropy_log.items()}
    print("\nper-query attention entropy (mean over samples):")
    print(f"{'subtask':>9}  " + "  ".join([f"Q{k}" for k in range(K)]))
    for s in target_subtasks:
        if s in entropy_means:
            row = "  ".join([f"{e:.2f}" for e in entropy_means[s]])
            print(f"{s:>9}  {row}")

    # save summary
    log_uniform = float(np.log(n_frames * H * W))
    print(f"\nuniform entropy upper bound (log {n_frames*H*W}) = {log_uniform:.3f}")
    (out_dir / "summary.json").write_text(json.dumps({
        "entropy_per_subtask_per_query": entropy_means,
        "uniform_entropy": log_uniform,
        "attention_pair_cosine": sim_log,
    }, indent=2))
    print(f"\n[saved] -> {out_dir}")


if __name__ == "__main__":
    main()
