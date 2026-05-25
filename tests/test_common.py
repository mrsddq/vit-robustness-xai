from models import load_model
from scripts.common import imagenet_transform, load_yaml, subgroup_gap


def test_config_loads():
    config = load_yaml("configs/robustness.yaml")

    assert config["model"]["name"] == "vit_b_16"
    assert len(config["evaluation"]["severity_levels"]) == 5


def test_subgroup_gap():
    assert round(subgroup_gap({"a": 0.90, "b": 0.80, "c": 0.85}), 4) == 0.10


def test_transform_is_composed():
    transform = imagenet_transform()

    assert hasattr(transform, "transforms")


def test_model_factory_rejects_unknown_model():
    try:
        load_model("not_a_model", pretrained=False)
    except ValueError as exc:
        assert "Unsupported torchvision model" in str(exc)
    else:
        raise AssertionError("Expected model factory to reject unknown models")
