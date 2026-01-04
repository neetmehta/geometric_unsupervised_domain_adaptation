import yaml
import torch
from types import SimpleNamespace


def dict_to_namespace(d):
    """Recursively convert dict to SimpleNamespace"""
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)


def resolve_device(device_str: str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "cuda":
        return torch.device("cuda")
    if device_str == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device option: {device_str}")


def load_config(config_path: str):
    """
    Loads YAML config and returns a dot-accessible config object.
    """
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # -------------------------
    # Basic validation
    # -------------------------
    required_sections = [
        "training",
        "model",
        "geometry",
        "dataset",
        "runtime",
    ]

    for section in required_sections:
        if section not in cfg_dict:
            raise KeyError(f"Missing config section: {section}")

    # -------------------------
    # Resolve device
    # -------------------------
    device = resolve_device(cfg_dict["runtime"].get("device", "auto"))
    cfg_dict["runtime"]["device"] = device

    # -------------------------
    # Convert to namespace
    # -------------------------
    cfg = dict_to_namespace(cfg_dict)

    return cfg
