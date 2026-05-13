"""기존 patch_embedding HDF5 에서 episode 당 max clip_id 만 뽑아 새 디렉토리로 복제.

save_dinov3_patch_repr.py 가 만든 multi-clip HDF5
    data/demo_N/clip_0..clip_7/{camera}: (n, H, W, D)
에서 각 demo 의 max clip_id 한 개만 골라서
    data/demo_N/clip_0/{camera}: (n, H, W, D)
로 재번호하여 저장. 결과는 "에피소드당 마지막 clip 만 가진 dataset" 으로,
contrastive_train.py 가 그대로 소비 가능 (filter flag 불필요).

사용 예:
    # 양 소스 last subset 동시 생성
    python script/make_lastclip_subset.py \
        --input data/patch_embeddings_libero_goal_hdf5 \
        --output data/patch_embeddings_libero_goal_hdf5_last

    python script/make_lastclip_subset.py \
        --input data/patch_embeddings_libero_goal_lerobot \
        --output data/patch_embeddings_libero_goal_lerobot_last
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
from tqdm import tqdm


def derive_one(h5_in_path: Path, h5_out_path: Path, overwrite: bool) -> tuple[int, int]:
    if h5_out_path.exists() and not overwrite:
        print(f"  [skip] {h5_out_path} exists (use --overwrite)")
        return 0, 0

    n_demos = 0
    n_skipped = 0
    with h5py.File(h5_in_path, "r") as fin, h5py.File(h5_out_path, "w") as fout:
        # top-level attrs 그대로 복사 + num_clips=1 로 갱신
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        fout.attrs["num_clips"] = 1
        fout.attrs["lastclip_derived_from"] = h5_in_path.name

        data_in = fin["data"]
        data_out = fout.create_group("data")
        demos = sorted(data_in.keys())
        for demo_key in demos:
            clip_keys = sorted(data_in[demo_key].keys())
            if not clip_keys:
                n_skipped += 1
                continue
            max_cid = max(int(ck.split("_")[1]) for ck in clip_keys)
            src_clip = data_in[f"{demo_key}/clip_{max_cid}"]

            demo_out = data_out.create_group(demo_key)
            clip_out = demo_out.create_group("clip_0")
            # clip-level attrs (frames_global, start_local 등) 보존 + 원본 id 기록
            for k, v in src_clip.attrs.items():
                clip_out.attrs[k] = v
            clip_out.attrs["original_clip_id"] = max_cid

            # 카메라 dataset 들을 그대로 복사 (h5py.copy 는 dtype/chunk/compression 보존)
            for cam in src_clip.keys():
                src_clip.copy(cam, clip_out)
            n_demos += 1

    return n_demos, n_skipped


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", type=Path, required=True,
                   help="patch_embedding 디렉토리 (예: data/patch_embeddings_libero_goal_hdf5)")
    p.add_argument("--output", type=Path, required=True,
                   help="새 디렉토리 (예: data/patch_embeddings_libero_goal_hdf5_last)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    in_dir = args.input
    out_dir = args.output
    if not in_dir.is_dir():
        raise SystemExit(f"input dir not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(in_dir.glob("*.hdf5"))
    if not h5_files:
        raise SystemExit(f"no *.hdf5 in {in_dir}")

    print(f"[in]  {in_dir}  ({len(h5_files)} hdf5)")
    print(f"[out] {out_dir}")

    total_demos = 0
    total_skipped = 0
    for h5_in in tqdm(h5_files, desc="tasks"):
        h5_out = out_dir / h5_in.name
        n, s = derive_one(h5_in, h5_out, args.overwrite)
        tqdm.write(f"  {h5_in.name}: demos={n}, skipped(empty)={s}")
        total_demos += n
        total_skipped += s

    print(f"[done] demos={total_demos} skipped={total_skipped}")


if __name__ == "__main__":
    main()
