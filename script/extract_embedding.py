"""
DINOv3 임베딩 추출 예제
- DINOv3 ViT 모델을 사용해 이미지의 feature embedding 추출
- CLS 토큰 임베딩과 patch 토큰 임베딩을 모두 보여줌
- 두 이미지 간 코사인 유사도 계산 예시도 포함

참고:
- facebook/dinov3-* 모델은 gated repo 입니다. 사용 전 HuggingFace에서 access 신청 필요.
  https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
- 토큰 로그인:  `huggingface-cli login`  또는 HF_TOKEN 환경변수
"""
import torch
import torch.nn.functional as F
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoImageProcessor, AutoModel


# ----------------------------- 설정 -----------------------------
MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"   # 가벼운 ViT-S/16
# 더 큰 모델 예시:
#   "facebook/dinov3-vitb16-pretrain-lvd1689m"   # ViT-B/16
#   "facebook/dinov3-vitl16-pretrain-lvd1689m"   # ViT-L/16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# 비교에 쓸 이미지 두 개 (인터넷에서 다운로드)
IMG_URLS = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    "https://images.cocodataset.org/val2017/000000039769.jpg",  # 고양이 두 마리
]


def load_image(src: str) -> Image.Image:
    """URL 또는 로컬 경로에서 이미지 로드"""
    if src.startswith("http"):
        resp = requests.get(src, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(src).convert("RGB")


def extract_embedding(model, processor, image: Image.Image):
    """
    이미지에서 DINOv3 임베딩 추출
    returns:
        cls_embedding:    (1, hidden_dim)  - 전체 이미지를 대표하는 벡터
        patch_embeddings: (1, n_patches, hidden_dim) - patch별 dense feature
    """
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

    # last_hidden_state: (B, 1 + n_register + n_patches, hidden_dim)
    last_hidden = outputs.last_hidden_state            # (1, N, D)
    cls_embedding = outputs.pooler_output              # (1, D) - CLS 토큰
    # DINOv3 ViT는 register 토큰을 사용하므로 patch만 떼어내려면 모델 config 확인 필요.
    # 간단히 last_hidden 전체를 dense feature로 사용해도 무방하지만,
    # 보통 첫 토큰(CLS) + register 토큰들을 제외한 나머지가 patch token입니다.
    num_reg = getattr(model.config, "num_register_tokens", 0)
    patch_embeddings = last_hidden[:, 1 + num_reg:, :]

    return cls_embedding, patch_embeddings


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    return float((a @ b.T).item())


def main():
    print(f"[INFO] Loading model: {MODEL_ID}")
    print(f"[INFO] Device: {DEVICE}, dtype: {DTYPE}")

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE).eval()

    embeddings = []
    for url in IMG_URLS:
        print(f"\n[INFO] Processing: {url}")
        image = load_image(url)
        print(f"  - image size : {image.size}")

        cls_emb, patch_emb = extract_embedding(model, processor, image)
        print(f"  - CLS embedding shape  : {tuple(cls_emb.shape)}")
        print(f"  - Patch embedding shape: {tuple(patch_emb.shape)}")
        print(f"  - CLS embedding (first 8 dims): {cls_emb[0, :8].float().cpu().numpy()}")

        embeddings.append(cls_emb)

    if len(embeddings) >= 2:
        sim = cosine_sim(embeddings[0], embeddings[1])
        print("\n" + "=" * 50)
        print(f"Cosine similarity between img0 and img1: {sim:.4f}")
        print("=" * 50)


if __name__ == "__main__":
    main()