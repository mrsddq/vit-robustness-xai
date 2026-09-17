"""Generate attention rollout heatmaps for ViT models."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.common import imagenet_transform, load_torchvision_model
from scripts.xai.gradcam import gradcam_heatmap


def attention_rollout(model, image_tensor):
    """Capture real per-head attention, including torchvision's tuple outputs."""
    backbone = getattr(model, "backbone", model)
    if not hasattr(backbone, "encoder"):
        raise ValueError("attention_rollout requires a ViT encoder")
    attentions, handles = [], []

    def request_weights(_module, args, kwargs):
        kwargs = dict(kwargs)
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return args, kwargs

    def collect_weights(_module, _args, output):
        if not isinstance(output, tuple) or output[1] is None:
            raise ValueError("Attention module did not return attention weights")
        attentions.append(output[1].detach())

    was_training = model.training
    try:
        model.eval()
        for block in backbone.encoder.layers:
            handles.append(block.self_attention.register_forward_pre_hook(request_weights, with_kwargs=True))
            handles.append(block.self_attention.register_forward_hook(collect_weights))
        with torch.no_grad():
            model(image_tensor.unsqueeze(0))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    if not attentions:
        raise ValueError("No attention layers were captured")
    tokens = attentions[0].shape[-1]
    identity = torch.eye(tokens, device=attentions[0].device, dtype=attentions[0].dtype)
    rollout = identity
    for attention in attentions:
        averaged = attention[0].mean(dim=0)
        residual = averaged + identity
        residual = residual / residual.sum(dim=-1, keepdim=True)
        rollout = residual @ rollout
    mask = rollout[0, 1:]
    size = int(mask.numel() ** 0.5)
    if size * size != mask.numel():
        raise ValueError("Patch tokens do not form a square image grid")
    mask = mask.reshape(size, size)
    maximum = mask.max()
    if maximum > 0:
        mask = mask / maximum
    return mask.cpu().numpy()


def save_heatmap(image_path, mask, output_path):
    import matplotlib.pyplot as plt
    from torchvision.transforms import CenterCrop, Resize
    # Match imagenet_transform geometric preprocessing, not a distorted full-image resize.
    with Image.open(image_path) as source:
        image = np.array(CenterCrop(224)(Resize(256)(source.convert("RGB"))))
    mask_resized = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((224, 224))) / 255.0

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(mask_resized, cmap="jet")
    axes[1].set_title("Attention map")
    axes[1].axis("off")
    axes[2].imshow(image)
    axes[2].imshow(mask_resized, cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main(model_name, method, inp, out_dir):
    input_dir = Path(inp)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_torchvision_model(model_name, pretrained=True).to(device).eval()
    transform = imagenet_transform()
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list(input_dir.rglob("*.jpg")) + list(input_dir.rglob("*.png"))
    for image_path in images[:20]:
        image_tensor = transform(Image.open(image_path).convert("RGB")).to(device)
        if method == "attention_rollout":
            mask = attention_rollout(model, image_tensor)
        elif method == "gradcam":
            mask = gradcam_heatmap(model, image_tensor.unsqueeze(0))
        else:
            raise ValueError(f"Unsupported XAI method: {method}")
        output_path = output_dir / f"{image_path.stem}_{method}.png"
        save_heatmap(str(image_path), mask, output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vit_b_16")
    parser.add_argument("--method", default="attention_rollout", choices=["attention_rollout", "gradcam"])
    parser.add_argument("--input", default="data/clean/")
    parser.add_argument("--output", default="outputs/heatmaps/")
    args = parser.parse_args()
    main(args.model, args.method, args.input, args.output)
