import yaml
import torch
from types import SimpleNamespace


def dict_to_namespace(d):
    """Recursively convert dictionary to SimpleNamespace for dot-accessible config.
    
    Enables accessing dictionary keys as attributes (e.g., cfg.training.batch_size
    instead of cfg['training']['batch_size']).
    
    Args:
        d (dict): Dictionary to convert, potentially with nested dictionaries.
    
    Returns:
        SimpleNamespace: Namespace object with all dict keys as attributes.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)


def resolve_device(device_str: str):
    """Resolve device string to torch.device object.
    
    Supports 'auto' for automatic selection (cuda if available, else cpu),
    and explicit 'cuda' or 'cpu' specification.
    
    Args:
        device_str (str): Device specification string ('auto', 'cuda', or 'cpu').
    
    Returns:
        torch.device: Resolved device object.
    
    Raises:
        ValueError: If device_str is not a recognized option.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "cuda":
        return torch.device("cuda")
    if device_str == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device option: {device_str}")


def load_config(config_path: str):
    """Load YAML configuration and return dot-accessible config object.
    
    Validates required configuration sections and resolves device settings.
    Converts configuration dictionary to SimpleNamespace for convenient attribute access.
    
    Args:
        config_path (str): Path to YAML configuration file.
    
    Returns:
        SimpleNamespace: Configuration object with dot-accessible attributes.
    
    Raises:
        KeyError: If required configuration sections are missing.
    
    Required sections: 'training', 'model', 'geometry', 'virtual_dataset', 'runtime'.
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
        "virtual_dataset",
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
