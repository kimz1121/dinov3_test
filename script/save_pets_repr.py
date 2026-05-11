"""평평한(또는 클래스 하위폴더) 이미지 폴더에서 DINOv3 임베딩을 추출해 HDF5 로 저장.

`save_dinov3_repr.py` 의 단순화 버전 — robocasa 의 demo/camera 4단 구조 대신
일반 분류용 데이터셋(고양이/강아지 등)에 맞는 평평한 스키마를 쓴다.

지원 입력 레이아웃
    (A) 클래스 하위폴더:
        data/pets/
          cats/  cat1.jpg cat2.jpg ...
          dogs/  dog1.jpg dog2.jpg ...

    (B) 평평 + 파일명 prefix (data/cat_vs_dog 처럼 `cat.123.jpg` / `dog.45.jpg`):
        data/cat_vs_dog/
          cat.0.jpg  cat.1.jpg ...  dog.0.jpg  dog.1.jpg ...
        → class = 파일명에서 첫 '.' 앞 토큰

    (C) 그 외 평평 폴더: 모두 class "all" 로 묶음

출력
    {output_root}/{name}__dinov3_{embedder}.hdf5
      attrs:
        model_id     : str
        embedder     : "cls" | "patch_mean"
        dim          : int   (D)
        num_images   : int   (N)
        classes      : (K,) str — 정렬된 unique 클래스명
      datasets:
        embeddings   : (N, D) float32, L2 정규화
        filenames    : (N,)  str (입력 폴더 기준 상대경로)
        class_ids    : (N,)  int32 — `classes` attr 의 인덱스

사용 예
    # cls + patch_mean 둘 다 만들기 (각각 따로 실행)
    python script/save_pets_repr.py data/cat_vs_dog --embedder cls
    python script/save_pets_repr.py data/cat_vs_dog --embedder patch_mean

    # 클래스 하위폴더 레이아웃
    python script/save_pets_repr.py data/pets --embedder cls --name pets
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from dinov3_embedders import DEFAULT_MODEL_ID, build_embedder


IMG_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def discover_images(root: Path) -> list[tuple[Path, str]]:
    """입력 폴더를 스캔해서 [(image_path, class_name), ...] 반환.

    - 하위폴더가 있고 그 안에 이미지가 있으면 (A) 모드: class = 하위폴더 이름
    - 평평 폴더면 (B/C) 모드: class = 파일명에서 첫 '.' 앞 토큰 (없으면 "all")
    """
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    has_class_subdirs = any(
        any(c.suffix.lower() in IMG_EXTS for c in d.iterdir() if c.is_file())
        for d in subdirs
    )

    pairs: list[tuple[Path, str]] = []
    if has_class_subdirs:
        for d in subdirs:
            cls = d.name
            for f in sorted(d.iterdir()):
                if f.is_file() and f.suffix.lower() in IMG_EXTS:
                    pairs.append((f, cls))
    else:
        for f in sorted(root.iterdir()):
            if not (f.is_file() and f.suffix.lower() in IMG_EXTS):
                continue
            # `cat.123.jpg` → stem_before_ext="cat.123" → cls="cat"
            # `cat.jpg`     → stem_before_ext="cat"     → cls="all" (확장자만 있는 경우)
            stem_before_ext = f.name.rsplit(".", 1)[0]
            cls = stem_before_ext.split(".", 1)[0] if "." in stem_before_ext else "all"
            pairs.append((f, cls))

    return sorted(pairs, key=lambda x: (x[1], x[0].name))


def encode_all(embedder, paths: list[Path], batch_size: int) -> np.ndarray:
    """이미지 경로 리스트 → (N, D) L2 정규화 numpy float32."""
    out: list[np.ndarray] = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        emb = embedder.encode_batch(imgs)        # (B, D) cpu float32
        out.append(emb.numpy())
        print(f"  encoded {min(i + batch_size, len(paths))}/{len(paths)}")
    return (
        np.concatenate(out, axis=0)
        if out
        else np.zeros((0, embedder.dim), np.float32)
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "input_root",
        type=Path,
        help="이미지 폴더 (평평하거나 클래스 하위폴더 구조)",
    )
    p.add_argument(
        "--embedder",
        choices=["cls", "patch_mean"],
        default="cls",
    )
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/embeddings_pets"),
    )
    p.add_argument(
        "--name",
        default=None,
        help="출력 파일명 prefix (기본: input_root 디렉토리 이름)",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_root.is_dir():
        raise SystemExit(f"[ERROR] not a directory: {args.input_root}")

    name = args.name or args.input_root.name
    out_path = args.output_root / f"{name}__dinov3_{args.embedder}.hdf5"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"[ERROR] {out_path} exists (use --overwrite)")

    pairs = discover_images(args.input_root)
    if not pairs:
        raise SystemExit(f"[ERROR] no images under {args.input_root}")

    paths = [p for p, _ in pairs]
    class_names_per_img = [c for _, c in pairs]
    classes = sorted(set(class_names_per_img))
    class_to_id = {c: i for i, c in enumerate(classes)}
    class_ids = np.array([class_to_id[c] for c in class_names_per_img], dtype=np.int32)
    rel_filenames = [str(p.relative_to(args.input_root)) for p in paths]

    print(f"[INFO] embedder={args.embedder}  model={args.model_id}")
    print(f"[INFO] input={args.input_root}  → {out_path}")
    print(f"[INFO] {len(paths)} images, {len(classes)} classes: {classes}")

    embedder = build_embedder(args.embedder, model_id=args.model_id)
    print(f"[INFO] dim={embedder.dim}  device={embedder.device}  dtype={embedder.dtype}")

    E = encode_all(embedder, paths, args.batch_size)
    assert E.shape == (len(paths), embedder.dim), E.shape

    str_dt = h5py.string_dtype()
    with h5py.File(out_path, "w") as f:
        f.attrs["model_id"] = embedder.model_id
        f.attrs["embedder"] = embedder.name
        f.attrs["dim"] = embedder.dim
        f.attrs["num_images"] = len(paths)
        f.attrs["classes"] = np.array(classes, dtype=str_dt)

        f.create_dataset("embeddings", data=E.astype(np.float32))
        f.create_dataset("filenames", data=np.array(rel_filenames, dtype=str_dt))
        f.create_dataset("class_ids", data=class_ids)

    print(f"[done] saved → {out_path}  ({len(paths)} × {embedder.dim}d)")


if __name__ == "__main__":
    main()
