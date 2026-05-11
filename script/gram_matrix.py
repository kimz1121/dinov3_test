"""
폴더 내 이미지들의 pairwise cosine similarity (gram matrix) 계산 + heatmap 시각화.

사용:
    python script/gram_matrix.py <folder> [--out PATH]

재사용 가능한 로직은 `script/similarity.py` 의 `ImageEmbedder` /
`gram_matrix` / `plot_heatmap` 으로 분리되어 있음. 이 파일은 CLI 진입점만.
"""
import argparse
from pathlib import Path

from similarity import (
    ImageEmbedder,
    format_gram,
    gram_matrix,
    plot_heatmap,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="이미지가 들어있는 폴더 경로")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("similarity_heatmap.png"),
        help="heatmap PNG 저장 경로 (default: similarity_heatmap.png)",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="bwr",
        help=(
            "matplotlib colormap 이름 (default: inferno). "
            "추천: inferno/magma/plasma/viridis (sequential), "
            "coolwarm/RdBu_r/bwr (diverging, 부호 의미 강조용)"
        ),
    )
    args = parser.parse_args()

    if not args.folder.is_dir():
        raise SystemExit(f"[ERROR] Not a directory: {args.folder}")

    print(f"[INFO] Loading model + scanning folder: {args.folder}")
    embedder = ImageEmbedder()
    paths, E = embedder.encode_folder(args.folder)

    if len(paths) == 0:
        raise SystemExit(f"[ERROR] No images found in {args.folder}")

    gram = gram_matrix(E)
    labels = [p.name for p in paths]

    print()
    print(format_gram(gram, labels))

    plot_heatmap(gram, labels, args.out, cmap=args.cmap)
    print(f"\n[INFO] Heatmap saved → {args.out}")


if __name__ == "__main__":
    main()
