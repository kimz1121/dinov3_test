"""
DINOv3 기반 이미지 임베딩 + 유사도 유틸 모듈
- ImageEmbedder: 모델 로딩/임베딩 추출 캡슐화 (재사용 가능)
- gram_matrix: 단위 벡터 stack → (N, N) cosine similarity matrix
- format_gram: 콘솔용 텍스트 포맷팅 (소수점 1자리)
- plot_heatmap: matplotlib heatmap PNG 저장
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


IMG_EXTS: set[str] = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
}


class ImageEmbedder:
    """DINOv3 CLS 임베딩 추출기.

    한 번 instantiate 해두고 여러 이미지를 인코딩할 때 재사용.
    출력 임베딩은 모두 L2 정규화된 단위 벡터(cpu 텐서).
    """

    def __init__(
        self,
        model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (
            torch.float16 if self.device == "cuda" else torch.float32
        )

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = (
            AutoModel.from_pretrained(model_id, dtype=self.dtype)
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def encode(self, image: Image.Image) -> torch.Tensor:
        """단일 PIL 이미지 → (1, D) L2 정규화된 CLS 임베딩 (cpu)."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        emb = out.pooler_output                       # (1, D)
        emb = F.normalize(emb.float(), dim=-1)        # 단위 벡터
        return emb.cpu()

    def encode_path(self, path: Path) -> torch.Tensor:
        """파일 경로 → (1, D)."""
        img = Image.open(path).convert("RGB")
        return self.encode(img)

    def encode_folder(
        self,
        folder: Path,
        exts: set[str] = IMG_EXTS,
        verbose: bool = True,
    ) -> tuple[list[Path], torch.Tensor]:
        """폴더 → (정렬된 이미지 경로 리스트, (N, D) 단위 벡터 텐서).

        하위 디렉터리는 탐색하지 않음 (1단계만).
        """
        paths = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )
        if not paths:
            return [], torch.empty(0)

        embs = []
        for p in paths:
            embs.append(self.encode_path(p))
            if verbose:
                print(f"  + {p.name}")
        E = torch.cat(embs, dim=0)                    # (N, D)
        return paths, E


def gram_matrix(E: torch.Tensor) -> np.ndarray:
    """단위 벡터 stack (N, D) → (N, N) cosine similarity matrix ∈ [-1, 1].

    입력이 L2 정규화되어 있다는 전제 — ImageEmbedder.encode* 의 출력이 그러함.
    """
    return (E @ E.T).numpy()


def format_gram(gram: np.ndarray, labels: list[str]) -> str:
    """Gram 행렬을 콘솔용 문자열로. 각 원소는 부호 포함 소수점 1자리."""
    n = len(labels)
    label_w = min(max((len(l) for l in labels), default=0), 24)
    short = [
        (l if len(l) <= label_w else l[: label_w - 1] + "…").ljust(label_w)
        for l in labels
    ]
    cell_w = 5

    lines = ["[Gram matrix — cosine similarity ∈ [-1, 1], 1 decimal]"]
    header = " " * (label_w + 2) + " ".join(
        f"{i:>{cell_w}}" for i in range(n)
    )
    lines.append(header)
    for i, (lbl, row) in enumerate(zip(short, gram)):
        cells = " ".join(f"{v:+.1f}".rjust(cell_w) for v in row)
        lines.append(f"{lbl}  {cells}    ({i})")
    return "\n".join(lines)


def plot_heatmap(
    gram: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str = "Pairwise image similarity (DINOv3 CLS)",
    # cmap: str = "coolwarm",
    cmap: str = "inferno",
    annotate: bool = True,
    boundaries: list[int] | None = None,
    max_fig_size: float = 16.0,
    max_tick_labels: int = 60,
) -> None:
    """Heatmap PNG 저장. vmin=-1, vmax=1, 큰 N 도 안전하게 처리.

    sequential cmap(inferno 등)에서도 가독성을 유지하기 위해
    각 셀의 배경 밝기(luminance)에 따라 텍스트 색을 자동 전환.

    큰 N 대응:
        - fig_size 는 `max_fig_size` 인치로 cap (작은 N 은 cell_size 비율 유지)
        - N > max_tick_labels 면 모든 label 그리는 대신 block 경계만 표기
        - N > 24 면 annotate 강제 비활성 (한 셀이 너무 작아 글자 못 들어감)
    """
    import matplotlib.pyplot as plt   # 모듈 임포트 비용 회피 (CLI에서만 호출됨)
    from matplotlib import colormaps

    n = len(labels)
    cell_size = 0.7
    fig_size = min(max_fig_size, max(5.0, cell_size * n + 2.0))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    vmin, vmax = -1.0, 1.0
    im = ax.imshow(gram, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    if n <= max_tick_labels:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
    elif boundaries:
        # 큰 N: block 경계 중앙에 block 대표 label 만 표기
        block_starts = [0] + list(boundaries) + [n]
        centers = [
            (block_starts[i] + block_starts[i + 1]) / 2 - 0.5
            for i in range(len(block_starts) - 1)
        ]
        # 각 block 첫 sample 의 label 을 대표로 사용 (보통 "{Task}/..." 형태라 task 이름이 앞)
        block_labels = [labels[block_starts[i]].split("/")[0] for i in range(len(block_starts) - 1)]
        ax.set_xticks(centers)
        ax.set_yticks(centers)
        ax.set_xticklabels(block_labels, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(block_labels, fontsize=10)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if n > 24:
        annotate = False

    if annotate:
        cm = colormaps.get_cmap(cmap)
        for i in range(n):
            for j in range(n):
                v = gram[i, j]
                # cell의 배경 RGB 의 perceived luminance 로 텍스트 색 결정
                r, g, b, _ = cm((v - vmin) / (vmax - vmin))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = "white" if lum < 0.5 else "black"
                ax.text(
                    j, i, f"{v:+.1f}",
                    ha="center", va="center", color=text_color, fontsize=9,
                )

    if boundaries:
        # 블록 사이에 흰색 구분선 — task/그룹 경계 시각화
        for b in boundaries:
            ax.axhline(b - 0.5, color="white", linewidth=1.5)
            ax.axvline(b - 0.5, color="white", linewidth=1.5)

    plt.colorbar(im, ax=ax, label="cosine similarity", shrink=0.8)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
