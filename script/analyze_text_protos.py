"""CLIP text prototype (13 sub-task) 시각화.

세 가지 단계 비교:
  (a) raw   : frozen CLIP encoder 의 512-d 출력 (학습 X)
  (b) infonce-projected: alignment_infonce 학습 후 text_proj_net 통과 128-d
  (c) anchor-projected : anchor_regression  학습 후 text_proj_net 통과 128-d

각 단계별로
  - 13x13 cosine Gram matrix heatmap
  - t-SNE 2D scatter (annotated with id)
를 2 행 x N 열 grid 에 한 figure 로 출력.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--text-protos",
        type=Path,
        default=Path("runs/exp2_cross_task_vl_alignment_infonce/text_protos.pt"),
        help="raw CLIP text proto (학습 시 저장된 .pt)",
    )
    p.add_argument(
        "--vl-ckpts",
        type=Path,
        nargs="*",
        default=[
            Path("runs/exp2_cross_task_vl_alignment_infonce/best.pt"),
            Path("runs/exp2_cross_task_vl_anchor_regression/best.pt"),
        ],
        help="trained text_proj_net 을 가진 vl_model ckpt 들",
    )
    p.add_argument(
        "--ckpt-names",
        type=str,
        nargs="*",
        default=["infonce-projected", "anchor-projected"],
        help="ckpt 컬럼 제목 (--vl-ckpts 와 같은 길이)",
    )
    p.add_argument(
        "--meta",
        type=Path,
        default=Path("data/patch_embeddings/_unified_subtasks.json"),
    )
    p.add_argument(
        "--instr-dir",
        type=Path,
        default=Path("data/patch_embeddings"),
        help="<Task>_instructions.json 위치 (대표 instruction 복원에 사용)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("runs/text_protos_analysis.png"),
    )
    p.add_argument("--perplexity", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_instruction_strs(meta: dict, instr_dir: Path) -> list[str]:
    """meta['mapping'] 을 기반으로 sub-task id -> 대표 instruction.

    contrastive_train_cross_task_vl.build_subtask_instruction_texts 와 동일 규칙
    (task 이름 알파벳 + local_ti 작은 순) 적용.
    """
    mapping = meta["mapping"]
    sid_candidates: dict[int, list[tuple[str, int]]] = {}
    for key, sid in mapping.items():
        task, local_ti_str = key.split("/")
        sid_candidates.setdefault(int(sid), []).append((task, int(local_ti_str)))

    sid_to_text: dict[int, str] = {}
    for sid, cands in sid_candidates.items():
        cands.sort()
        task, local_ti = cands[0]
        instr_path = instr_dir / f"{task}_instructions.json"
        instr_meta = json.loads(instr_path.read_text())
        sid_to_text[sid] = instr_meta["instructions"][str(local_ti)]
    return [sid_to_text[i] for i in sorted(sid_to_text.keys())]


def plot_gram(ax, gram: np.ndarray, n: int, title: str) -> None:
    im = ax.imshow(gram, cmap="bwr", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(n), fontsize=8)
    ax.set_yticklabels(range(n), fontsize=8)
    ax.set_xlabel("sub-task id", fontsize=9)
    ax.set_ylabel("sub-task id", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    for i in range(n):
        for j in range(n):
            v = gram[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6, color=("white" if abs(v) > 0.7 else "black"))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine sim")


def plot_tsne(ax, Z: np.ndarray, n: int, perplexity: float, seed: int, title: str) -> None:
    perp = min(perplexity, max(2.0, (n - 1) / 3))
    Z2 = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
    ).fit_transform(Z)
    cmap = plt.get_cmap("tab20")
    for i in range(n):
        ax.scatter(Z2[i, 0], Z2[i, 1], s=140, color=cmap(i % 20),
                   edgecolors="black", linewidths=0.8)
        ax.annotate(str(i), (Z2[i, 0], Z2[i, 1]),
                    ha="center", va="center", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(alpha=0.3)


def load_text_proj_state(ckpt_path: Path) -> dict:
    """vl_model ckpt 에서 text_proj_net 만 추출."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "vl_model" in ckpt:
        sd = ckpt["vl_model"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    text_sd = {k[len("text_proj_net."):]: v for k, v in sd.items()
               if k.startswith("text_proj_net.")}
    if not text_sd:
        raise SystemExit(f"no text_proj_net.* keys in {ckpt_path}: have {list(sd.keys())[:6]}...")
    return text_sd


def project_text(text_protos: np.ndarray, text_proj_state: dict) -> np.ndarray:
    """text_proj_net (Linear-GELU-Linear-LayerNorm) 의 sequential 인덱스 0/2/3 으로 직접 복원."""
    import torch.nn.functional as F

    W0 = text_proj_state["0.weight"]; b0 = text_proj_state["0.bias"]
    W2 = text_proj_state["2.weight"]; b2 = text_proj_state["2.bias"]
    Wn = text_proj_state["3.weight"]; bn = text_proj_state["3.bias"]
    x = torch.from_numpy(text_protos.astype(np.float32))
    h = F.linear(x, W0, b0)
    h = F.gelu(h)
    h = F.linear(h, W2, b2)
    h = F.layer_norm(h, Wn.shape, weight=Wn, bias=bn)
    h = F.normalize(h, dim=-1)
    return h.numpy().astype(np.float32)


def summarize(Z: np.ndarray, name: str, instruction_strs: list[str]) -> None:
    gram = Z @ Z.T
    n = Z.shape[0]
    off = gram[~np.eye(n, dtype=bool)]
    iu = np.triu_indices(n, k=1)
    pairs = sorted(
        [(gram[i, j], i, j) for i, j in zip(*iu)],
        key=lambda x: -x[0],
    )
    print(f"\n--- {name} ---")
    print(f"  off-diag cosine: mean={off.mean():.4f}  "
          f"min={off.min():.4f}  max={off.max():.4f}  "
          f"median={np.median(off):.4f}")
    print(f"  top-5 most similar pairs:")
    for v, i, j in pairs[:5]:
        print(f"    cos={v:.4f}  {i:>2} vs {j:>2}  "
              f"({instruction_strs[i][:34]!r:36s} ~ {instruction_strs[j][:34]!r})")


def main() -> None:
    args = parse_args()

    text_protos = torch.load(args.text_protos, map_location="cpu")
    if hasattr(text_protos, "numpy"):
        Z_raw = text_protos.numpy().astype(np.float32)
    else:
        Z_raw = np.asarray(text_protos, dtype=np.float32)
    n, d = Z_raw.shape
    print(f"[INFO] raw text_protos shape=({n}, {d})")
    Z_raw = Z_raw / np.clip(np.linalg.norm(Z_raw, axis=1, keepdims=True), 1e-8, None)

    meta = json.loads(args.meta.read_text())
    instruction_strs = build_instruction_strs(meta, args.instr_dir)
    assert len(instruction_strs) == n

    columns = [("raw CLIP (512-d)", Z_raw)]
    for ckpt_path, name in zip(args.vl_ckpts, args.ckpt_names):
        if not ckpt_path.exists():
            print(f"[WARN] missing ckpt: {ckpt_path} — skip")
            continue
        text_sd = load_text_proj_state(ckpt_path)
        Z_proj = project_text(Z_raw, text_sd)
        dim = Z_proj.shape[1]
        columns.append((f"{name} ({dim}-d)", Z_proj))

    for label, Z in columns:
        summarize(Z, label, instruction_strs)

    ncol = len(columns)
    fig = plt.figure(figsize=(6.0 * ncol, 11))
    gs = fig.add_gridspec(
        2, ncol,
        height_ratios=[1.0, 1.0],
        wspace=0.25, hspace=0.25,
        left=0.05, right=0.97, top=0.92, bottom=0.30,
    )

    for col_idx, (label, Z) in enumerate(columns):
        gram = Z @ Z.T
        off = gram[~np.eye(n, dtype=bool)]
        mean_off = off.mean()

        ax_g = fig.add_subplot(gs[0, col_idx])
        plot_gram(ax_g, gram, n,
                  title=f"Gram — {label}\n(mean off-diag cos={mean_off:.3f})")

        ax_t = fig.add_subplot(gs[1, col_idx])
        plot_tsne(ax_t, Z, n, perplexity=args.perplexity, seed=args.seed,
                  title=f"t-SNE — {label}")

    fig.suptitle(
        "Text prototype geometry: raw CLIP vs contrastive-trained projections",
        fontsize=14, fontweight="bold", y=0.97,
    )

    legend_lines = []
    for i, s in enumerate(instruction_strs):
        short = textwrap.shorten(s, width=90, placeholder="...")
        legend_lines.append(f"{i:>2}  =  {short}")
    legend_text = "\n".join(legend_lines)
    fig.text(0.5, 0.02, legend_text, ha="center", va="bottom",
             fontsize=8, family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="none"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"\n[done] saved -> {args.out}")


if __name__ == "__main__":
    main()
