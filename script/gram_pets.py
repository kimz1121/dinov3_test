"""`save_pets_repr.py` 가 저장한 평평한 HDF5 임베딩으로 Gram matrix + heatmap.

`gram_from_hdf5.py` 의 평평 폴더용 버전 — robocasa 의 demo/camera 구조 대신
(embeddings, filenames, class_ids) 스키마를 읽는다.

핵심:
- 1개 파일 → heatmap 1장
- 2개 파일 → 좌우 subplot 으로 동일 데이터셋의 cls vs patch_mean 같은 비교용
- 행/열은 (class_id, filename) 으로 정렬 → 클래스가 블록처럼 모임
- 클래스 경계에 흰 구분선

사용 예
    # cls 와 patch_mean 임베딩 각각 저장된 두 파일을 한 PNG 로 비교
    python script/gram_pets.py \
        data/embeddings_pets/cat_vs_dog__dinov3_cls.hdf5 \
        data/embeddings_pets/cat_vs_dog__dinov3_patch_mean.hdf5 \
        --out gram_pets_cls_vs_patch.png

    # 단일 파일만
    python script/gram_pets.py \
        data/embeddings_pets/cat_vs_dog__dinov3_cls.hdf5 \
        --out gram_pets_cls.png

    # 라벨에 파일명까지 (기본은 클래스명만)
    python script/gram_pets.py ... --label-mode filename
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def read_h5_pets(h5_path: Path) -> dict:
    """평평 임베딩 HDF5 → dict(E, filenames, class_ids, classes, attrs)."""
    with h5py.File(h5_path, "r") as f:
        E = f["embeddings"][()].astype(np.float32)
        filenames = [s.decode() if isinstance(s, bytes) else s
                     for s in f["filenames"][()]]
        class_ids = f["class_ids"][()].astype(np.int64)
        classes = [s.decode() if isinstance(s, bytes) else s
                   for s in f.attrs["classes"]]
        meta = {
            "embedder": f.attrs.get("embedder", "?"),
            "dim": int(f.attrs.get("dim", E.shape[1])),
            "model_id": f.attrs.get("model_id", "?"),
        }
    return {
        "E": E,
        "filenames": filenames,
        "class_ids": class_ids,
        "classes": classes,
        "meta": meta,
    }


def sort_by_class(data: dict) -> tuple[np.ndarray, list[str], list[int]]:
    """class_id 후 filename 순으로 정렬.

    Returns:
        E_sorted   : (N, D)
        labels     : 각 행 라벨 (class/filename)
        boundaries : 클래스가 바뀌는 인덱스 (그룹 구분선 위치)
    """
    n = len(data["filenames"])
    order = sorted(
        range(n),
        key=lambda i: (int(data["class_ids"][i]), data["filenames"][i]),
    )
    E = data["E"][order]
    sorted_ids = data["class_ids"][order]
    sorted_files = [data["filenames"][i] for i in order]

    boundaries: list[int] = []
    for i in range(1, n):
        if sorted_ids[i] != sorted_ids[i - 1]:
            boundaries.append(i)

    labels_class = [data["classes"][cid] for cid in sorted_ids]
    return E, labels_class, sorted_files, boundaries


def gram_cosine(E: np.ndarray) -> np.ndarray:
    """L2 정규화된 임베딩 가정 → E @ E.T 가 cosine sim."""
    return E @ E.T


def plot_grid(
    grams: list[np.ndarray],
    labels: list[str],
    titles: list[str],
    out_path: Path,
    cmap: str,
    annotate: bool,
    boundaries: list[int],
) -> None:
    """1~2 개의 Gram heatmap 을 한 figure 에 그린다.

    1개면 단일 axes, 2개면 좌우 subplot. y축 라벨은 첫 axes 에만.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    n = len(labels)
    k = len(grams)
    cell = 0.7
    per_w = max(5.0, cell * n + 2.0)
    fig_w = per_w * k + 1.5
    fig_h = max(5.0, cell * n + 2.5)

    fig, axes = plt.subplots(1, k, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[0]

    vmin, vmax = -1.0, 1.0
    cm = colormaps.get_cmap(cmap)

    for ax, g, title in zip(axes, grams, titles):
        im = ax.imshow(g, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=11)

        if annotate:
            for i in range(n):
                for j in range(n):
                    v = g[i, j]
                    r, gc, b, _ = cm((v - vmin) / (vmax - vmin))
                    lum = 0.299 * r + 0.587 * gc + 0.114 * b
                    color = "white" if lum < 0.5 else "black"
                    ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                            color=color, fontsize=7)

        for b_idx in boundaries:
            ax.axhline(b_idx - 0.5, color="white", linewidth=1.5)
            ax.axvline(b_idx - 0.5, color="white", linewidth=1.5)

        fig.colorbar(im, ax=ax, label="cosine similarity", shrink=0.7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "h5_files",
        type=Path,
        nargs="+",
        help="save_pets_repr.py 출력 HDF5 (1~2개; 2개면 side-by-side 비교)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("gram_pets.png"),
    )
    p.add_argument(
        "--cmap",
        default="bwr",
        help="matplotlib colormap (default bwr — 부호 강조)",
    )
    p.add_argument(
        "--label-mode",
        choices=["class", "filename", "both"],
        default="class",
        help="행/열 라벨 (class | filename | both)",
    )
    p.add_argument(
        "--annotate-threshold",
        type=int,
        default=30,
        help="N ≤ 이 값일 때만 셀에 숫자 오버레이",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.h5_files) > 2:
        raise SystemExit("[ERROR] 최대 2 개 파일까지 지원합니다.")

    datasets = []
    for h5 in args.h5_files:
        if not h5.is_file():
            raise SystemExit(f"[ERROR] not a file: {h5}")
        datasets.append(read_h5_pets(h5))
        print(
            f"[INFO] {h5.name}: N={len(datasets[-1]['filenames'])}, "
            f"D={datasets[-1]['E'].shape[1]}, "
            f"embedder={datasets[-1]['meta']['embedder']}"
        )

    # 2개 파일을 비교하려면 같은 데이터셋(같은 N 과 같은 파일명 순서)이어야 의미가 있음
    if len(datasets) == 2:
        a, b = datasets
        if a["filenames"] != b["filenames"] or list(a["class_ids"]) != list(b["class_ids"]):
            raise SystemExit(
                "[ERROR] 두 파일의 (filenames, class_ids) 가 다릅니다. "
                "같은 입력 폴더로 만든 두 embedder 결과여야 비교 가능."
            )

    # 정렬은 첫 파일 기준 — 두 번째 파일도 동일 순서 적용
    E0, labels_class, filenames_sorted, boundaries = sort_by_class(datasets[0])
    n = E0.shape[0]

    # label 모드
    if args.label_mode == "class":
        labels = labels_class
    elif args.label_mode == "filename":
        labels = filenames_sorted
    else:  # both
        labels = [f"{c}/{Path(f).stem}" for c, f in zip(labels_class, filenames_sorted)]

    grams = [gram_cosine(E0)]
    titles = [f"{args.h5_files[0].stem}\n(embedder={datasets[0]['meta']['embedder']})"]

    if len(datasets) == 2:
        # 두 번째 데이터셋도 동일 순서로 재정렬 — filenames 가 일치하므로
        # datasets[0] 의 order 를 그대로 적용
        order_lookup = {fn: i for i, fn in enumerate(datasets[1]["filenames"])}
        order = [order_lookup[fn] for fn in filenames_sorted]
        E1 = datasets[1]["E"][order]
        grams.append(gram_cosine(E1))
        titles.append(f"{args.h5_files[1].stem}\n(embedder={datasets[1]['meta']['embedder']})")

    annotate = n <= args.annotate_threshold
    print(f"[INFO] N={n}, classes(boundary at row): {boundaries}, annotate={annotate}")

    plot_grid(
        grams=grams,
        labels=labels,
        titles=titles,
        out_path=args.out,
        cmap=args.cmap,
        annotate=annotate,
        boundaries=boundaries,
    )
    print(f"[done] saved → {args.out}")


if __name__ == "__main__":
    main()
