"""DINOv3 임베더 — 두 가지 pooling 방식을 같은 인터페이스로 제공.

- CLSEmbedder      : pooler_output (CLS 토큰) — 기존 similarity.ImageEmbedder 와 동치
- PatchMeanEmbedder: multi-scale(224, 448) patch token 평균 — Lotus save_dinov2_repr.py 스타일

둘 다 (B, D) L2 정규화된 cpu float32 텐서를 반환한다.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


DEFAULT_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"


class BaseDinoEmbedder(Protocol):
    """공통 인터페이스 — script 에서 인스턴스만 바꿔서 호출할 수 있게."""

    name: str       # "cls" | "patch_mean" — 출력 경로에 쓰임
    dim: int        # 임베딩 차원 (D)

    def encode_batch(self, imgs: list[Image.Image]) -> torch.Tensor:
        """(B,) PIL 이미지 리스트 → (B, D) L2 정규화 cpu float32."""
        ...


def _load_model(model_id: str, device: str, dtype: torch.dtype):
    """processor + eval 모드 모델 로드. 두 embedder 가 공유."""
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return processor, model


class CLSEmbedder:
    """`pooler_output` 기반 CLS 임베딩.

    Processor 가 알아서 224 리사이즈 + 노멀라이즈 하므로 우리는 그대로 위임.
    similarity.ImageEmbedder 와 동일한 결과를 내야 한다(회귀 테스트로 검증).
    """

    name = "cls"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (
            torch.float16 if self.device == "cuda" else torch.float32
        )
        self.processor, self.model = _load_model(model_id, self.device, self.dtype)
        self.dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode_batch(self, imgs: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        emb = out.pooler_output                                # (B, D)
        emb = F.normalize(emb.float(), dim=-1)
        return emb.cpu()


class PatchMeanEmbedder:
    """Multi-scale (224, 448) patch token mean pool.

    Lotus `save_dinov2_repr.py` 의 `process_images` 를 DINOv3 로 포팅:
        1. 입력 이미지를 두 스케일로 cv2-equivalent resize (PIL.LANCZOS)
        2. 각 스케일에서 model forward → last_hidden_state 의 patch token 추출
           (DINOv3 는 [CLS] + register 4개 + patches 순서)
        3. patch token 을 (B, h, w, D) 로 rearrange 후 bilinear interpolate 로
           max_size = max(sizes) // patch_size 로 통일 (224/16=14, 448/16=28 → 28)
        4. 두 스케일 평균
        5. 공간축 mean → (B, D), 마지막에 L2 정규화

    reference 와의 의도적 차이:
        - patch_size 가 14(DINOv2) → 16(DINOv3) 이라 max_size 가 32 → 28
        - reference 의 수동 normalize 대신 processor 의 normalize 만 위임
          (do_resize=False 로 리사이즈는 우리가 직접 제어)
    """

    name = "patch_mean"
    SIZES: tuple[int, ...] = (448, 224)

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        sizes: tuple[int, ...] = SIZES,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (
            torch.float16 if self.device == "cuda" else torch.float32
        )
        self.processor, self.model = _load_model(model_id, self.device, self.dtype)

        cfg = self.model.config
        self.dim = int(cfg.hidden_size)
        self.patch_size = int(cfg.patch_size)
        self.num_register_tokens = int(getattr(cfg, "num_register_tokens", 0))

        # 모든 size 가 patch_size 의 배수여야 한다 — 아니면 ViT 입력으로 못 씀
        for s in sizes:
            if s % self.patch_size != 0:
                raise ValueError(
                    f"size {s} not divisible by patch_size {self.patch_size}"
                )
        self.sizes = tuple(sizes)
        self.max_size = max(sizes) // self.patch_size   # 공통 공간 해상도

    def _forward_at_size(
        self, imgs_resized: list[Image.Image], size: int
    ) -> torch.Tensor:
        """주어진 정사각 해상도에서 patch feature map 반환.

        Returns:
            (B, max_size, max_size, D) float32 cpu? — gpu 위에 둠
        """
        # do_resize=False 로 리사이즈는 우리가 미리 처리한 상태를 그대로 사용
        inputs = self.processor(
            images=imgs_resized,
            do_resize=False,
            do_center_crop=False,
            return_tensors="pt",
        ).to(self.device)

        out = self.model(**inputs)
        last_hidden = out.last_hidden_state                   # (B, 1+R+N, D)
        # CLS + register 토큰 제거 → patch 만 남김
        patches = last_hidden[:, 1 + self.num_register_tokens :, :]   # (B, N, D)

        h = size // self.patch_size
        if patches.shape[1] != h * h:
            raise RuntimeError(
                f"patch count {patches.shape[1]} != {h*h} at size {size}"
            )

        # (B, N, D) → (B, D, h, h) → bilinear → (B, D, max, max) → (B, max, max, D)
        feat = rearrange(patches, "b (h w) d -> b d h w", h=h, w=h).float()
        feat = F.interpolate(
            feat,
            size=(self.max_size, self.max_size),
            mode="bilinear",
            align_corners=True,
            antialias=True,
        )
        feat = rearrange(feat, "b d h w -> b h w d")
        return feat

    @staticmethod
    def _resize_pil(img: Image.Image, size: int) -> Image.Image:
        # cv2.INTER_LINEAR 와 가장 유사한 PIL 옵션은 BILINEAR.
        # reference 는 cv2.resize (default INTER_LINEAR) 였음.
        return img.convert("RGB").resize((size, size), Image.BILINEAR)

    @torch.no_grad()
    def encode_batch(self, imgs: list[Image.Image]) -> torch.Tensor:
        feats_per_scale: list[torch.Tensor] = []
        for size in self.sizes:
            imgs_resized = [self._resize_pil(im, size) for im in imgs]
            feats_per_scale.append(self._forward_at_size(imgs_resized, size))

        # 스케일 평균 → 공간 평균 → (B, D)
        stacked = torch.stack(feats_per_scale, dim=0)          # (S, B, h, w, D)
        feat = stacked.mean(dim=0)                              # (B, h, w, D)
        emb = feat.mean(dim=(1, 2))                             # (B, D)
        emb = F.normalize(emb, dim=-1)
        return emb.cpu()


class PatchSpatialEmbedder:
    """Spatial patch feature map — mean pooling 이전 단계까지.

    PatchMeanEmbedder 와 동일한 multi-scale 로직을 거치되, 마지막 `(B, h, w, D)`
    feature map 을 그대로 반환한다 (공간 평균 X, L2 정규화 X). Contrastive 학습에
    토큰 단위로 들어가는 입력을 만들기 위함.

    기본은 단일 스케일 224 → (14, 14, D). 멀티스케일이 필요하면 sizes 로 지정.
    """

    name = "patch_spatial"
    SIZES: tuple[int, ...] = (224,)

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        sizes: tuple[int, ...] = SIZES,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (
            torch.float16 if self.device == "cuda" else torch.float32
        )
        self.processor, self.model = _load_model(model_id, self.device, self.dtype)

        cfg = self.model.config
        self.dim = int(cfg.hidden_size)
        self.patch_size = int(cfg.patch_size)
        self.num_register_tokens = int(getattr(cfg, "num_register_tokens", 0))

        for s in sizes:
            if s % self.patch_size != 0:
                raise ValueError(
                    f"size {s} not divisible by patch_size {self.patch_size}"
                )
        self.sizes = tuple(sizes)
        self.max_size = max(sizes) // self.patch_size

    def _forward_at_size(
        self, imgs_resized: list[Image.Image], size: int
    ) -> torch.Tensor:
        inputs = self.processor(
            images=imgs_resized,
            do_resize=False,
            do_center_crop=False,
            return_tensors="pt",
        ).to(self.device)

        out = self.model(**inputs)
        last_hidden = out.last_hidden_state
        patches = last_hidden[:, 1 + self.num_register_tokens :, :]

        h = size // self.patch_size
        if patches.shape[1] != h * h:
            raise RuntimeError(
                f"patch count {patches.shape[1]} != {h*h} at size {size}"
            )

        feat = rearrange(patches, "b (h w) d -> b d h w", h=h, w=h).float()
        feat = F.interpolate(
            feat,
            size=(self.max_size, self.max_size),
            mode="bilinear",
            align_corners=True,
            antialias=True,
        )
        feat = rearrange(feat, "b d h w -> b h w d")
        return feat

    @staticmethod
    def _resize_pil(img: Image.Image, size: int) -> Image.Image:
        return img.convert("RGB").resize((size, size), Image.BILINEAR)

    @torch.no_grad()
    def encode_batch(self, imgs: list[Image.Image]) -> torch.Tensor:
        feats_per_scale: list[torch.Tensor] = []
        for size in self.sizes:
            imgs_resized = [self._resize_pil(im, size) for im in imgs]
            feats_per_scale.append(self._forward_at_size(imgs_resized, size))

        stacked = torch.stack(feats_per_scale, dim=0)        # (S, B, h, w, D)
        feat = stacked.mean(dim=0)                            # (B, h, w, D)
        return feat.cpu()


_EMBEDDERS: dict[str, type] = {
    "cls": CLSEmbedder,
    "patch_mean": PatchMeanEmbedder,
    "patch_spatial": PatchSpatialEmbedder,
}


def build_embedder(name: str, **kwargs) -> BaseDinoEmbedder:
    """'cls' | 'patch_mean' 문자열로 embedder 인스턴스 생성."""
    try:
        cls = _EMBEDDERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown embedder '{name}'. Choose from: {sorted(_EMBEDDERS)}"
        )
    return cls(**kwargs)
