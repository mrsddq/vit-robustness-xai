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
