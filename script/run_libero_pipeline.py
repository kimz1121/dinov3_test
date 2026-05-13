"""LIBERO-goal 두 소스(원본 HDF5 / LeRobot 포팅) 풀 파이프라인 오케스트레이터.

각 소스에 대해 sequential 로:
    1) clip 추출 (extract_libero_{hdf5,lerobot}_clips.py)
    2) DINOv3 patch embedding 저장 (save_dinov3_patch_repr.py)
    3) Contrastive 학습 (contrastive_train.py)
    4) Eval (contrastive_eval.py)
끝에 compare_libero_runs.py 로 두 결과를 묶어 비교 아티팩트 생성.

산출 디렉토리:
    runs/libero_goal_pair_{ts}/
      index.json
      hdf5.log, lerobot.log
      compare/    (compare_libero_runs.py 가 채움)

사용 예:
    # smoke (2 task × 5 ep × 3 epoch)
    python script/run_libero_pipeline.py --smoke

    # 풀 런
    python script/run_libero_pipeline.py
    python script/run_libero_pipeline.py --sources hdf5
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = REPO_ROOT / "script"

SOURCES = {
    "hdf5": {
        "extract_script": "extract_libero_hdf5_clips.py",
        "clips_dir": "data/libero_goal_hdf5_clips",
        "patch_dir": "data/patch_embeddings_libero_goal_hdf5",
    },
    "lerobot": {
        "extract_script": "extract_libero_lerobot_clips.py",
        "clips_dir": "data/libero_goal_lerobot_clips",
        "patch_dir": "data/patch_embeddings_libero_goal_lerobot",
    },
}

# Smoke 용 task 2개 (양쪽 소스에 동일 존재 — task_index 10, 16)
SMOKE_TASKS = ["put_the_bowl_on_the_plate", "turn_on_the_stove"]


def check_disk_free(min_gb: int = 20) -> None:
    usage = shutil.disk_usage(REPO_ROOT)
    free_gb = usage.free / (1024**3)
    print(f"[disk] free={free_gb:.1f} GB (요구={min_gb} GB)")
    if free_gb < min_gb:
        raise SystemExit(f"disk free {free_gb:.1f} GB < {min_gb} GB. 정리 후 재시도")


def run(cmd: list[str], log_path: Path | None = None) -> None:
    print(f"\n$ {' '.join(cmd)}")
    if log_path is None:
        rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
    else:
        with open(log_path, "ab") as lf:
            lf.write(f"\n\n$ {' '.join(cmd)}\n".encode())
            lf.flush()
            proc = subprocess.Popen(
                cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
                lf.write(line)
                lf.flush()
            rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"command failed (rc={rc}): {' '.join(cmd)}")


def run_one_source(
    src: str,
    args: argparse.Namespace,
    pair_dir: Path,
    timestamp: str,
) -> Path:
    """한 소스에 대한 4 단계 실행. 학습 run dir 경로 반환."""
    cfg = SOURCES[src]
    log_path = pair_dir / f"{src}.log"
    clips_dir = REPO_ROOT / cfg["clips_dir"]
    patch_dir = REPO_ROOT / cfg["patch_dir"]
    run_dir = REPO_ROOT / "runs" / f"libero_goal_{src}_{timestamp}"

    print(f"\n{'='*70}\n[{src}] 시작\n{'='*70}")

    # 1) extract clips
    extract_cmd = [
        sys.executable, str(SCRIPT_DIR / cfg["extract_script"]),
        "--output", str(clips_dir),
        "--num-clips", str(args.num_clips),
        "--clip-length", str(args.clip_length),
        "--sampling-mode", args.sampling_mode,
        "--seed", str(args.seed),
    ]
    if args.smoke:
        extract_cmd += ["--tasks", *SMOKE_TASKS, "--max-episodes", str(args.smoke_eps)]
    if args.tasks:
        extract_cmd += ["--tasks", *args.tasks]
    if args.max_episodes is not None:
        extract_cmd += ["--max-episodes", str(args.max_episodes)]
    if args.overwrite:
        extract_cmd += ["--overwrite"]
    run(extract_cmd, log_path)

    # 2) patch embeddings
    patch_cmd = [
        sys.executable, str(SCRIPT_DIR / "save_dinov3_patch_repr.py"),
        "--manifest", str(clips_dir / "manifest.json"),
        "--png-root", str(clips_dir),
        "--output-root", str(patch_dir),
        "--batch-size", str(args.embed_batch_size),
    ]
    if args.overwrite:
        patch_cmd += ["--overwrite"]
    run(patch_cmd, log_path)

    # 3) train
    train_cmd = [
        sys.executable, str(SCRIPT_DIR / "contrastive_train.py"),
        "--patch-h5-root", str(patch_dir),
        "--epochs", str(args.epochs),
        "--warmup-epochs", str(args.warmup_epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--out-dir", str(run_dir),
        "--log-every", str(args.log_every),
    ]
    run(train_cmd, log_path)

    # 4) eval
    eval_cmd = [
        sys.executable, str(SCRIPT_DIR / "contrastive_eval.py"),
        "--run-dir", str(run_dir),
    ]
    run(eval_cmd, log_path)

    print(f"\n[{src}] 완료 → {run_dir}")
    return run_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sources", nargs="+", default=["hdf5", "lerobot"],
                   choices=list(SOURCES.keys()))
    p.add_argument("--smoke", action="store_true",
                   help=f"빠른 통합 검증: 2 task × {{smoke-eps}} ep × 3 epoch")
    p.add_argument("--smoke-eps", type=int, default=5)

    # 공통 (clip 추출)
    p.add_argument("--num-clips", type=int, default=8)
    p.add_argument("--clip-length", type=int, default=4)
    p.add_argument("--sampling-mode", choices=["uniform", "random"], default="uniform")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")

    # patch embed
    p.add_argument("--embed-batch-size", type=int, default=32)

    # train
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=50)

    p.add_argument("--min-disk-gb", type=int, default=20)
    p.add_argument("--skip-compare", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # smoke 일 때 학습 파라미터 강제 축소
    if args.smoke:
        args.epochs = 3
        args.warmup_epochs = 1
        args.batch_size = 32
        args.embed_batch_size = 16
        args.log_every = 5
        args.num_clips = max(4, args.num_clips // 2)
        args.min_disk_gb = 5

    check_disk_free(args.min_disk_gb)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pair_dir = REPO_ROOT / "runs" / f"libero_goal_pair_{ts}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pair-dir] {pair_dir}")
    print(f"[args] {vars(args)}")

    run_dirs: dict[str, str] = {}
    for src in args.sources:
        rd = run_one_source(src, args, pair_dir, ts)
        run_dirs[src] = str(rd)

    index = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "smoke": args.smoke,
        "sources": run_dirs,
        "args": vars(args),
    }
    with open(pair_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"\n[index] → {pair_dir / 'index.json'}")

    # 5) compare (양쪽 다 있을 때만)
    if not args.skip_compare and len(run_dirs) >= 2:
        compare_cmd = [
            sys.executable, str(SCRIPT_DIR / "compare_libero_runs.py"),
            "--pair-dir", str(pair_dir),
        ]
        run(compare_cmd)

    print(f"\n[done] pair_dir={pair_dir}")


if __name__ == "__main__":
    main()
