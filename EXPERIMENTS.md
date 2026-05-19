# DINOv3 Test — 실험 정리

`dinov3_test/` 에서 진행한 실험들의 전체 요약. 큰 줄기는 **DINOv3 patch
representation 위에서 task / nuisance(카메라) disentanglement contrastive
모델을 학습·평가**하는 것이고, 데이터 소스로 LIBERO-Goal(HDF5/LeRobot)
과 RoboCasa 를 사용한다. 그 외에 데이터 준비, 단순 분류/유사도 sanity
check, 시각화 보조 실험이 함께 들어 있다.

## 0. 왜 이 실험을 했나 (Motivation)

문제의식은 한 줄로 요약된다:

> **"로봇 manipulation task 들은 시각적으로 너무 비슷해서, pretrained vision
> backbone (DINOv3) 의 raw feature 만 봐서는 task 를 구분할 수 없다.
> contrastive 로 더 짜내봤자 끝까지 안 풀리는 케이스가 정말 있나?"**

세부 동기는 세 단계로 정리할 수 있다.

1. **Raw DINOv3 가 task 를 거의 구분 못한다는 것**부터 보여야 한다.
   `eval_summary.json` 의 `gram_separation.raw_dinov3_task.separation`
   값들이 그 증거다 — LIBERO/RoboCasa 6개 run 전부에서 **0.013 ~ 0.035**
   사이로, within/cross 가 거의 같다 (≈ 0.82 vs 0.80). 즉 같은 task 의
   두 프레임 ↔ 다른 task 의 두 프레임이 cosine 공간에서 사실상
   구분되지 않는다. `gram_test_dinov3_raw.png` 가 같은 그림을 시각적으로
   보여주는 패널.

2. **그 위에 contrastive head 를 학습시키면 어디까지 분리되는가**.
   `contrastive_train.py` 가 attention pool + (z_task / z_nuis) 두 head
   를 SupCon + Orth 손실로 학습한다. `gram_separation.z_task_task` 가
   raw 0.01–0.03 → 학습 후 0.29–**1.10** 까지 올라가는지가 첫 번째
   확인 포인트.

3. **그래도 안 풀리는 케이스가 있는가** — 학습 후에도 task 가 섞이는
   "hard" 케이스를 실제 이미지로 끄집어내서 보여주는 것이 이 프로젝트의
   원래 목표. `script/visualize_hard_cases.py` 와 각 run 의
   `hard_cases/hard_confused_cases.png`, `easy_separated_cases.png` 가
   그 산출물이다. **`task_on_ztask` 가 1.0 이 아니면 = 학습 후에도
   못 푸는 케이스가 남아 있다는 뜻**이고, 그 케이스를 시각적으로 들여다
   보면 "두 task 의 시각 신호가 본질적으로 같았다"는 근거가 된다.

이 모티베이션이 실험 설계의 세 축을 만들었다:

- **데이터 소스 두 종류** (LIBERO-Goal vs RoboCasa) — task 간 시각 유사성
  스펙트럼이 정반대인 두 환경을 비교.
- **클립 범위 (full vs last-clip-only)** — full 프레임을 다 쓰면
  task-무관 노이즈가 들어가니, "마지막 프레임만" 으로 task signal 을
  최대치로 만들었을 때 어디까지 분리되는지를 본다.
- **포맷 변환의 영향** (HDF5 vs LeRobot, LIBERO 만 해당) — 같은 데이터를
  포맷만 바꿔도 학습 결과가 흔들리는지 sanity.

## 1. 환경

- Docker: `nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04` + PyTorch 2.7.0
  (cu126). `transformers>=4.56,<5`, `huggingface_hub<1`, `lerobot`.
- 컨테이너에서 호스트의 `~/code/dinov3_test` 를 `/home/iw/dinov3_test`
  로 마운트. GPU 는 RTX 4070 Ti 기준.
- `docker-compose.yaml`, `docker/dockerfile`, `.env` 참고.

## 2. 데이터 파이프라인

raw 시뮬레이션 데이터 → clip PNG → DINOv3 patch HDF5 의 2단 추출.

### 2-1. clip 추출 (`script/extract_*`)

| 스크립트 | 입력 | 출력 |
|---|---|---|
| `extract_robocasa_clips.py` / `extract_robocasa_images.py` | RoboCasa HDF5 | `data/robocasa_clips/`, `data/robocasa_start*` PNG |
| `extract_libero_hdf5_clips.py` | LIBERO-Goal HDF5 (Lotus 포맷) | `data/libero_goal_hdf5_clips/` |
| `extract_libero_lerobot_clips.py` | LIBERO-Goal LeRobot 포맷 | `data/libero_goal_lerobot_clips/` |
| `lerobot_to_libero.py` | LeRobot → Lotus 스타일 HDF5 변환 |  |
| `make_lastclip_subset.py` | 위 clip 들 중 episode 마지막 clip만 필터링 (last-frame 실험용) |  |

각 task 당 episode → clip(여러 frame) → camera view 별 PNG 로 저장하고
manifest 를 남긴다.

### 2-2. 임베딩 추출 (`script/save_*`, `dinov3_embedders.py`)

- `save_dinov3_repr.py` — DINOv3 CLS / patch-mean 임베딩, (n, D) 로
  평탄화. 빠른 sanity / similarity 용도.
- `save_dinov3_patch_repr.py` — **공간 구조 유지** 한 patch token
  `(n, H=14, W=14, D=384)` 을 `data/patch_embeddings*/{Task}.hdf5` 로
  저장. contrastive 학습의 입력 토큰.
- `save_pets_repr.py` — Oxford-IIIT Pets 의 CLS/patch 임베딩 추출.
  Gram matrix 비교 실험용.
- `reference/save_dinov2_repr.py` — 원래 Lotus 의 DINOv2 구현, 포팅
  기준이 된 참고 파일.

생성된 HDF5 디렉토리:

- `data/patch_embeddings/` — RoboCasa full-clip
- `data/patch_embeddings_last/` — RoboCasa last-clip only
- `data/patch_embeddings_libero_goal_hdf5{,_last}/`
- `data/patch_embeddings_libero_goal_lerobot{,_last}/`

## 3. 핵심 모델 — task / nuisance disentanglement

`script/contrastive_train.py`

- 입력: `{Task}.hdf5` 의 clip 당 `(n, 14, 14, 384)` patch token.
- 모델: 단일 attention pool (num_queries 개의 learned query) → shared
  trunk → 두 projection head `z_task`, `z_nuisance`.
- 손실: `SupCon(z_task, task_label) + λ_n · SupCon(z_nuis, camera_label)
  + λ_o · Orth(z_task, z_nuis)`.
- 분할: task 별 episode 를 정렬 후 앞 80% train / 뒤 20% test.

기본 하이퍼파라미터 (모든 run 동일):

```
batch_size=128, epochs=60, warmup=5, lr=1e-3, wd=1e-4,
temperature=0.1, num_queries=4, d_task=128, d_nuis=64,
hidden=512, num_heads=8, dropout=0.1,
lambda_nuis=1.0, lambda_ortho=0.005, seed=42
```

평가 (`script/contrastive_eval.py`):

- Test split 의 `dinov3_raw / z_task / z_nuis` 각각에 대해 Gram
  matrix, k-NN accuracy, UMAP (task 색 / 카메라 색) 시각화.
- 결과는 `<run>/eval/` 아래 `gram_test_*.png`, `umap_*.png`,
  `eval_summary.json` 으로 저장.

오케스트레이터: `run_libero_pipeline.py`, `run_all_sweeps.sh`,
`sweep_num_queries.sh`.

## 4. 학습 / 평가 Run 카탈로그 (`runs/`)

| Run | 데이터 | 비고 |
|---|---|---|
| `20260511_171650` | RoboCasa 3 task | epoch 5 sanity |
| `20260511_171734` | RoboCasa 3 task | epoch 60 + full eval, 초기 베이스라인 |
| `libero_goal_hdf5_20260512_035025` | LIBERO-Goal HDF5, 10 task, agentview+eye_in_hand | 60 epoch |
| `libero_goal_hdf5_last_20260512_041940` | 위 + last-clip only | last-frame 효과 비교 |
| `libero_goal_lerobot_20260512_035025` | LIBERO-Goal LeRobot, 10 task | 데이터 출처 비교 |
| `libero_goal_lerobot_last_20260512_041940` | 위 + last-clip only |  |
| `libero_goal_pair_last_20260512_041940` | HDF5 vs LeRobot 페어 비교 | `compare_libero_runs.py` 용 |
| `robocasa_6task_full_20260512_052054` | RoboCasa 6 task, 3 camera | full-clip |
| `robocasa_6task_last_20260512_052054` | 위 + last-clip only |  |
| `robocasa_6task_pair_20260512_053948` | full vs last 페어 비교 |  |
| `all_sweeps_20260513_072342/` | 4개 데이터셋 × `num_queries ∈ {4,8,16}` | epoch 30, K=쿼리 수 sweep |

각 run 폴더에는 `config.json`, `task_to_id`, `ckpt_epoch_*.pt`,
`best.pt`, `log.jsonl`, `eval/` 가 들어 있다.

대표 지표 (예시):

- `libero_goal_hdf5_20260512_035025`: task gram separation ≈ 0.285,
  task k-NN accuracy ≈ 0.787.
- `robocasa_6task_full_20260512_052054`: task gram separation ≈ 0.345,
  k-NN accuracy ≈ 0.839.

## 5. 주제별 실험 묶음

위 run 들이 답하려는 질문은 대체로 다음 4가지다.

1. **데이터 소스 차이** — 동일한 LIBERO-Goal task 라도 HDF5 (Lotus)
   vs LeRobot 포맷에서 학습/평가 결과가 같은가? → `libero_goal_hdf5_*`
   vs `libero_goal_lerobot_*` 페어 + `libero_goal_pair_last_*` +
   `compare_libero_runs.py` (Procrustes 정렬 비교).
2. **클립 범위(full vs last-frame)** — clip 전체 vs 에피소드 마지막
   clip 만 사용했을 때 task/nuisance 분리력이 어떻게 바뀌나? →
   `*_last_*` run + `make_lastclip_subset.py`.
3. **Attention pool 쿼리 수 (K)** — `num_queries` 를 4/8/16 으로
   바꿔가며 disentanglement 품질의 sweep. → `all_sweeps_20260513_*` +
   `sweep_num_queries.sh`.
4. **데이터셋 일반화** — RoboCasa(주방 조작) vs LIBERO-Goal(가정 조작)
   에서 같은 모델이 어떻게 동작하나? → 위 두 도메인 run 비교.

## 6. 결과 분석 — LIBERO vs RoboCasa

각 run 의 `eval/eval_summary.json` 과 pair run 의 `compare/SUMMARY.md`
에서 핵심 지표만 모은 것이다.

### 6-1. Raw DINOv3 의 task 분리력 (학습 전 베이스라인)

`gram_separation.raw_dinov3_task.separation` (within − cross, 클수록↑):

| Run | within | cross | **separation** |
|---|---:|---:|---:|
| LIBERO HDF5 full | 0.834 | 0.818 | **0.016** |
| LIBERO HDF5 last | 0.841 | 0.805 | **0.035** |
| LIBERO LeRobot full | 0.834 | 0.820 | **0.014** |
| LIBERO LeRobot last | 0.831 | 0.799 | **0.032** |
| RoboCasa full | 0.752 | 0.739 | **0.013** |
| RoboCasa last | 0.750 | 0.736 | **0.014** |

→ **두 도메인 모두 raw DINOv3 는 task 를 거의 구분 못 한다** (separation
≈ 0). 동기 1번이 데이터로 확인됨.

### 6-2. Contrastive 학습 후 z_task 의 task 분리력

| Run | gram z_task **sep** | knn `task_on_ztask` | knn `cam_on_ztask` (작을수록↑) |
|---|---:|---:|---:|
| LIBERO HDF5 full | 0.285 | 0.787 | 0.979 |
| LIBERO HDF5 **last** | **1.099** | **1.000** | 0.920 |
| LIBERO LeRobot full | 0.310 | 0.795 | 0.974 |
| LIBERO LeRobot **last** | **1.090** | **1.000** | 0.883 |
| RoboCasa full | 0.345 | 0.839 | 0.818 |
| RoboCasa **last** | **0.454** | **0.879** | 0.760 |

→ 학습 후 분리도가 raw 대비 폭발적으로 향상 (LIBERO last 는 +30배 가까이).
**동기 2번 확인.**

### 6-3. LIBERO last-clip 페어 비교 (HDF5 vs LeRobot)

`runs/libero_goal_pair_last_20260512_041940/compare/SUMMARY.md` 의 핵심:

| metric | hdf5 | lerobot | diff |
|---|---:|---:|---:|
| `knn.task_on_ztask` | **1.000** | **1.000** | 0 |
| gram z_task separation | 1.099 | 1.090 | −0.009 |
| `knn.cam_on_ztask` (task에 카메라 누설) | 0.920 | **0.883** | −0.037 |
| `knn.task_on_znuis` (nuis에 task 누설) | 0.968 | **0.706** | **−0.262** |
| `disentanglement.D_mean` | 0.056 | **0.206** | +0.150 |

해석:
- **task 분리 자체는 두 소스 동일** (`task_on_ztask = 1.0`, gram sep ≈ 1.09).
  즉 LIBERO 는 last-clip 만 쓰면 어느 포맷이든 contrastive head 가 task
  를 완벽히 분리한다.
- **단, disentanglement 품질은 LeRobot 이 훨씬 좋다** — LeRobot 의 z_nuis 가
  task 정보를 덜 들고 있음 (0.706 vs 0.968). 두 포맷이 cover 하는 물리
  시간 (HDF5 20Hz vs LeRobot 10Hz) 과 no-op 필터링 차이 때문으로 추정.
- **포맷 변환 자체는 task 학습 결과를 흔들지 않는다**는 sanity 가 확보됨
  (동기 1·2 의 결과는 포맷 artifact 가 아니다).

### 6-4. RoboCasa full vs last 페어 비교

`runs/robocasa_6task_pair_20260512_053948/compare/SUMMARY.md` 핵심:

| metric | full | last | diff |
|---|---:|---:|---:|
| `knn.task_on_ztask` | 0.839 | **0.879** | +4%p |
| gram z_task separation | 0.345 | **0.454** | +31% |
| `knn.cam_on_ztask` | 0.818 | **0.760** | −6%p ✓ |
| `knn.task_on_znuis` (작을수록↑) | **0.676** | 0.798 | +12%p ⚠ |
| `disentanglement.D_mean` | 0.157 | 0.147 | ~동일 |
| best test loss | 8.93 | 5.75 | −36% |

해석:
- last 가 task 분리를 약간 더 잘 하지만 (+4%p), **LIBERO 만큼 극적이지는
  않다** (LIBERO 는 +21%p, +277%). RoboCasa 는 task 마다 초기 주방
  배치/물체가 이미 달라서 full clip 도 충분히 task signal 을 잡고 있음.
- 부작용: last 에서 z_nuis 에 task 정보 누설 증가 → full clip 이 들고 있던
  프레임 다양성이 일종의 nuisance regularization 역할을 했던 것으로 보임.

### 6-5. 종합 — 가설의 결론

| | LIBERO-Goal (10 task) | RoboCasa (6 task) |
|---|---|---|
| task 간 초기상태 | 거의 동일 | 이미 시각적으로 다름 |
| raw DINOv3 sep | ≈ 0.02 | ≈ 0.01 |
| full clip 후 `task_on_ztask` | 0.79 | 0.84 |
| **last clip 후 `task_on_ztask`** | **1.000** | **0.879** |
| last 의 분리 이득 | +21%p, gram +277% | +4%p, gram +31% |
| 학습 후에도 못 푸는 케이스? | **없음** (k-NN 100%) | **있음** (12% 오분류) |

**가설 정리** (사용자 가설 그대로):

- ✅ "task 간 visual 피처가 비슷하다" — raw DINOv3 separation ≈ 0 으로 확인.
- ✅ "raw 로는 구분이 안 된다" — 동일.
- ✅ "contrastive 로 학습시켰음에도 못 푸는 케이스가 있다" — **RoboCasa
  에서 확실히 관찰됨** (last clip 에서도 task k-NN 87.9%, gram sep 0.45
  로 LIBERO 의 1.0 / 1.10 에 크게 못 미침). 못 푼 케이스의 실체는 각
  run 의 `hard_cases/hard_confused_cases.png` 로 시각화되어 있다.
- ↔ 단, LIBERO 는 last clip 만 쓰면 100% 분리되어 "못 풀리는 케이스"
  주장이 성립하지 않음. **RoboCasa 가 가설의 핵심 evidence 역할**을 함.

가설의 메시지를 더 강하게 만들고 싶다면:
(i) RoboCasa 의 `hard_confused_cases.png` 에서 어떤 task 쌍이 섞이는지
정량화 (confusion matrix), (ii) last + 더 강한 nuisance penalty 로
"이론적 상한" 까지 밀어붙여도 RoboCasa 가 100% 에 못 닿는지 확인.

## 7. 보조 / sanity 실험

- `classify_image.py`, `extract_embedding.py`, `test.py` — 단일 이미지
  분류 및 임베딩 추출 데모.
- `similarity.py` — CLS vs patch-mean 임베딩의 cosine similarity 비교
  (`heatmap_cls.png`, `heatmap_patch.png`, `heatmap_all_*.png` 생성).
- `gram_matrix.py`, `gram_from_hdf5.py`, `gram_pets.py` — DINOv3 raw
  feature 의 Gram matrix 시각화. Oxford Pets 실험 결과는
  `gram_pets_cls_vs_patch.png`.
- `visualize_hard_cases.py` — k-NN 으로 잘 / 잘못 분리된 케이스의
  원본 이미지를 시각화.
- `draw_model_arch.py` → `figures/model_architecture.png`.
- Cat vs Dog (`data/cat_vs_dog*`), RoboCasa start-frame
  (`data/robocasa_start*`) 등 시각화 전용 미니 데이터셋.

## 8. 산출물 위치 빠른 참조

- 학습 코드 / 평가 코드: `script/contrastive_{train,eval}.py`
- 학습 결과 / 체크포인트: `runs/<exp>/`
- 평가 시각화: `runs/<exp>/eval/*.png`, `eval_summary.json`
- 임베딩 HDF5: `data/patch_embeddings*`, `data/embeddings*`
- 루트의 `heatmap_*.png`, `gram_pets_cls_vs_patch.png` 는 sanity
  실험 산출물.
