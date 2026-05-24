"""Generate attention rollout heatmaps for ViT models."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from scripts.common import imagenet_transform, load_torchvision_model


def attention_rollout(model, image_tensor):
    attentions = []
    hooks = []

    def hook(_module, _inputs, output):
        attentions.append(output.detach())

    for block in model.encoder.layers:
        hooks.append(block.self_attention.register_forward_hook(hook))

    with torch.no_grad():
        model(image_tensor.unsqueeze(0))

    for hook_handle in hooks:
        hook_handle.remove()

    rollout = torch.eye(attentions[0].shape[-1])
    for attention in attentions:
        attention_average = attention.squeeze(0).mean(0)
        rollout = torch.matmul(attention_average + torch.eye(attention_average.shape[-1]), rollout)

    mask = rollout[0, 1:]
    size = int(mask.shape[0] ** 0.5)
    return mask.reshape(size, size).numpy()


def save_heatmap(image_path, mask, output_path):
    image = np.array(Image.open(image_path).resize((224, 224)))
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
    if method != "attention_rollout":
        raise NotImplementedError("Only attention_rollout is implemented in this repository.")

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
        mask = attention_rollout(model, image_tensor)
        output_path = output_dir / f"{image_path.stem}_{method}.png"
        save_heatmap(str(image_path), mask, output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vit_b_16")
    parser.add_argument("--method", default="attention_rollout", choices=["attention_rollout"])
    parser.add_argument("--input", default="data/clean/")
    parser.add_argument("--output", default="outputs/heatmaps/")
    args = parser.parse_args()
    main(args.model, args.method, args.input, args.output)
