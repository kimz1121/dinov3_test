"""
DINOv3 이미지 분류 예제
- DINOv3는 self-supervised backbone 이므로 기본적으로 classification head가 없습니다.
- 따라서 일반적인 사용 방법은 두 가지:
    (A) Few-shot / kNN classification: 클래스별 reference 이미지의 임베딩을 모은 뒤
        쿼리 이미지의 임베딩과 cosine similarity 로 가장 가까운 클래스를 찾는 방식
    (B) Linear probe: backbone을 frozen 시키고 위에 작은 linear head를 fine-tune

이 스크립트는 (A) zero-train kNN 방식을 보여줍니다.
인터넷에서 받은 작은 샘플 이미지로 cat / dog 분류를 시연합니다.
"""
import torch
import torch.nn.functional as F
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoImageProcessor, AutoModel


MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# 클래스별 reference 이미지 (각 클래스의 prototype 을 만드는 데 사용)
REFERENCE_IMAGES = {
    "cat": [
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png",
    ],
    "dog": [
        "https://images.dog.ceo/breeds/husky/n02110185_10047.jpg",
        "https://images.dog.ceo/breeds/pug/n02110958_15626.jpg",
    ],
}

# 분류해 볼 쿼리 이미지들 (정답은 사람이 보고 판단)
QUERY_IMAGES = [
    ("dog?", "https://images.dog.ceo/breeds/beagle/n02088364_16065.jpg"),
    ("cat?", "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg"),
]


def load_image(src: str) -> Image.Image:
    if src.startswith("http"):
        resp = requests.get(src, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(src).convert("RGB")


@torch.no_grad()
def encode(model, processor, image: Image.Image) -> torch.Tensor:
    """이미지 → CLS 임베딩 (정규화된 벡터)"""
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    out = model(**inputs)
    emb = out.pooler_output                       # (1, D)
    emb = F.normalize(emb.float(), dim=-1)        # 코사인 유사도용 정규화
    return emb.cpu()


def build_prototypes(model, processor):
    """클래스별 reference 이미지들의 임베딩 평균을 prototype 으로 사용"""
    prototypes = {}
    for cls_name, urls in REFERENCE_IMAGES.items():
        embs = []
        print(f"\n[REF] class='{cls_name}'")
        for url in urls:
            img = load_image(url)
            emb = encode(model, processor, img)
            embs.append(emb)
            print(f"  + {url}  shape={tuple(emb.shape)}")
        proto = torch.cat(embs, dim=0).mean(dim=0, keepdim=True)
        proto = F.normalize(proto, dim=-1)
        prototypes[cls_name] = proto
    return prototypes


def classify(query_emb: torch.Tensor, prototypes: dict):
    """쿼리 임베딩과 각 prototype 의 cosine similarity 비교"""
    scores = {cls: float((query_emb @ proto.T).item())
              for cls, proto in prototypes.items()}
    best = max(scores, key=scores.get)
    return best, scores


def main():
    print(f"[INFO] Loading model: {MODEL_ID}")
    print(f"[INFO] Device: {DEVICE}, dtype: {DTYPE}")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE).eval()

    print("\n[STEP 1] Building class prototypes from reference images...")
    prototypes = build_prototypes(model, processor)

    print("\n[STEP 2] Classifying query images...")
    print("=" * 70)
    for label_hint, url in QUERY_IMAGES:
        img = load_image(url)
        emb = encode(model, processor, img)
        pred, scores = classify(emb, prototypes)

        score_str = ", ".join(f"{c}={s:+.4f}" for c, s in scores.items())
        print(f"[hint={label_hint:>6}]  pred = {pred:>3}   ({score_str})")
        print(f"   url: {url}")
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()