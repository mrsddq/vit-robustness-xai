"""Generate Grad-CAM / attention rollout heatmaps for ViT.
Usage: python scripts/xai/generate_heatmaps.py --method gradcam --input data/clean/ --output outputs/heatmaps/
"""
import argparse, torch
import torchvision.models as models
import torchvision.transforms as T
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                       T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])


def attention_rollout(model, img_tensor):
    """Simplified attention rollout for ViT."""
    attentions = []
    hooks = []
    def hook(module, input, output):
        attentions.append(output.detach())
    for block in model.encoder.layers:
        hooks.append(block.self_attention.register_forward_hook(hook))
    with torch.no_grad():
        model(img_tensor.unsqueeze(0))
    for h in hooks:
        h.remove()
    rollout = torch.eye(attentions[0].shape[-1])
    for attn in attentions:
        attn_avg = attn.squeeze(0).mean(0)
        rollout = torch.matmul(attn_avg + torch.eye(attn_avg.shape[-1]), rollout)
    mask = rollout[0, 1:]
    size = int(mask.shape[0] ** 0.5)
    return mask.reshape(size, size).numpy()


def save_heatmap(img_path, mask, out_path):
    img = np.array(Image.open(img_path).resize((224, 224)))
    mask_resized = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((224, 224))) / 255.0
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(mask_resized, cmap="jet"); axes[1].set_title("Attention map"); axes[1].axis("off")
    axes[2].imshow(img); axes[2].imshow(mask_resized, cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main(model_name, method, inp, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(models, model_name)(pretrained=True).to(device).eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = list(Path(inp).rglob("*.jpg")) + list(Path(inp).rglob("*.png"))
    for img_path in images[:20]:
        img_tensor = transform(Image.open(img_path).convert("RGB")).to(device)
        if method == "attention_rollout":
            mask = attention_rollout(model, img_tensor)
        else:
            print(f"Method {method}: install grad-cam library and use pytorch_grad_cam.GradCAM")
            continue
        save_heatmap(str(img_path), mask, out_dir / (img_path.stem + f"_{method}.png"))
        print(f"  Saved: {img_path.stem}_{method}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vit_b_16")
    p.add_argument("--method", default="attention_rollout", choices=["gradcam","attention_rollout","shap"])
    p.add_argument("--input", default="data/clean/")
    p.add_argument("--output", default="outputs/heatmaps/")
    a = p.parse_args()
    main(a.model, a.method, a.input, a.output)
