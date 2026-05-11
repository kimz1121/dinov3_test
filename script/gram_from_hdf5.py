"""save_dinov3_repr.py 가 저장한 HDF5 임베딩으로 Gram matrix + heatmap 생성.

HDF5 구조 (save_dinov3_repr.py 의 출력):
    data/
      demo_000/
        {camera_name}    shape (D,) float32, L2 normalized
        ...

여러 파일을 한 번에 받을 수 있고, 카메라 필터와 episode 개수 제한으로
heatmap 가독성을 조절한다. 임베딩은 이미 L2 정규화되어 저장돼 있어
`E @ E.T` 가 곧 cosine similarity matrix.

사용 예
    # 한 파일, 카메라 한 종류만, 처음 20 episode
    python script/gram_from_hdf5.py \
        data/embeddings/dinov3_cls/start/CheesyBread.hdf5 \
        --cameras robot0_agentview_left --max-episodes 20

    # 두 task 를 한 plot 으로 비교
    python script/gram_from_hdf5.py \
        data/embeddings/dinov3_cls/start/CheesyBread.hdf5 \
        data/embeddings/dinov3_cls/start/CloseCabinet.hdf5 \
        --cameras robot0_agentview_left --max-episodes 10 \
        --out gram_cls_two_tasks.png

    # 같은 데이터로 cls 와 patch_mean 비교 (각각 따로 실행 후 두 PNG 비교)
    python script/gram_from_hdf5.py \
        data/embeddings/dinov3_patch_mean/start/CheesyBread.hdf5 \
        --cameras robot0_agentview_left --max-episodes 20 \
        --out gram_patch_mean.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from similarity import format_gram, gram_matrix, plot_heatmap


# 카메라명을 라벨에 짧게 넣기 위한 축약 (사람이 읽기 쉽게)
CAMERA_ABBREV = {
    "robot0_agentview_left": "L",
    "robot0_agentview_right": "R",
    "robot0_eye_in_hand": "H",
}


def short_cam(cam: str) -> str:
    return CAMERA_ABBREV.get(cam, cam)


def read_h5_embeddings(
    h5_path: Path,
    cameras: list[str] | None,
    max_episodes: int | None,
) -> tuple[np.ndarray, list[str]]:
    """한 HDF5 파일에서 (필터된) 임베딩 + 라벨을 평탄화해 반환.

    Returns:
        E      : (N, D) float32 — 이미 L2 정규화돼 있음
        labels : 길이 N — "{file_stem}/ep{ep:03d}/{cam_abbrev}"
    """
    file_stem = h5_path.stem
    embeddings: list[np.ndarray] = []
    labels: list[str] = []

    with h5py.File(h5_path, "r") as f:
        all_cams: list[str] = list(f.attrs["cameras"])
        cams = cameras or all_cams
        unknown = [c for c in cams if c not in all_cams]
        if unknown:
            raise SystemExit(
                f"[ERROR] {h5_path}: unknown camera(s) {unknown}. "
                f"Available: {all_cams}"
            )

        demos = sorted(f["data"].keys())
        if max_episodes is not None:
            demos = demos[:max_episodes]

        for demo_key in demos:
            ep_idx = int(demo_key.split("_")[1])
            for cam in cams:
                ds = f[f"data/{demo_key}/{cam}"]
                embeddings.append(ds[()].astype(np.float32))
                labels.append(f"{file_stem}/ep{ep_idx:03d}/{short_cam(cam)}")

    E = np.stack(embeddings, axis=0) if embeddings else np.zeros((0, 0), np.float32)
    return E, labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "h5_files",
        type=Path,
        nargs="+",
        help="save_dinov3_repr.py 가 만든 HDF5 파일 1개 이상",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="포함할 카메라 이름 (기본: 파일 attrs 의 모든 카메라)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="각 파일당 사용할 episode 수 제한 (큰 N 은 heatmap 가독성 떨어짐)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("gram_heatmap.png"),
        help="heatmap PNG 저장 경로",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="bwr",
        help="matplotlib colormap (default: bwr — diverging, 부호 강조)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="heatmap 제목 (기본: 자동 생성)",
    )
    parser.add_argument(
        "--annotate-threshold",
        type=int,
        default=24,
        help="N <= 이 값일 때만 셀에 숫자 오버레이 (기본 24)",
    )
    parser.add_argument(
        "--print-matrix",
        action="store_true",
        help="콘솔에도 Gram 텍스트 출력 (N 작을 때만 권장)",
    )
    args = parser.parse_args()

    # --- 모든 파일에서 임베딩 모으기 ---
    all_E: list[np.ndarray] = []
    all_labels: list[str] = []
    boundaries: list[int] = []   # 다음 파일이 시작하는 행/열 인덱스
    cursor = 0
    for h5 in args.h5_files:
        if not h5.is_file():
            raise SystemExit(f"[ERROR] not a file: {h5}")
        E, labels = read_h5_embeddings(h5, args.cameras, args.max_episodes)
        if E.size == 0:
            print(f"[warn] no embeddings from {h5}, skipping")
            continue
        print(f"[INFO] {h5}: {E.shape}  ({len(labels)} labels)")
        all_E.append(E)
        all_labels.extend(labels)
        cursor += len(labels)
        boundaries.append(cursor)

    if not all_E:
        raise SystemExit("[ERROR] No embeddings loaded.")

    E = np.concatenate(all_E, axis=0)
    print(f"[INFO] combined: {E.shape}  N={len(all_labels)}")

    # 마지막 boundary 는 매트릭스 끝이라 선을 그릴 필요 없음
    block_lines = boundaries[:-1] if len(boundaries) >= 2 else None

    # similarity.gram_matrix 는 torch tensor 받음 → 변환
    gram = gram_matrix(torch.from_numpy(E))

    n = len(all_labels)
    annotate = n <= args.annotate_threshold

    if args.print_matrix:
        if n <= 50:
            print()
            print(format_gram(gram, all_labels))
        else:
            print(f"[skip] --print-matrix: N={n} > 50, omitting text output")

    title = args.title or (
        f"Gram (cosine) — N={n}, files={len(args.h5_files)}, "
        f"cams={'/'.join(args.cameras) if args.cameras else 'all'}"
    )
    plot_heatmap(
        gram,
        all_labels,
        args.out,
        title=title,
        cmap=args.cmap,
        annotate=annotate,
        boundaries=block_lines,
    )
    print(
        f"\n[INFO] Heatmap saved → {args.out}  "
        f"(annotate={annotate}, block_lines={block_lines})"
    )


if __name__ == "__main__":
    main()
