from transformers import pipeline


pipe = pipeline(
    task="image-feature-extraction",
    model="facebook/dinov3-vits16-pretrain-lvd1689m",
)

pipe("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")