"""Vision Transformer model factory."""
import torchvision.models as models


def load_model(name="vit_b_16", pretrained=True, num_classes=1000):
    """Load a torchvision ViT-style model with a consistent factory interface."""
    if not hasattr(models, name):
        raise ValueError(f"Unsupported torchvision model: {name}")

    constructor = getattr(models, name)
    if pretrained:
        weights_enum = getattr(models, f"{name.upper()}_Weights", None)
        weights = weights_enum.DEFAULT if weights_enum else "DEFAULT"
        return constructor(weights=weights)

    return constructor(weights=None, num_classes=num_classes)
