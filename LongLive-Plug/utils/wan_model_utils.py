"""Base-model path handling for the two LongLive-Plug backbones."""
import os
from utils.config import wan_default_config


def resolve_wan_model_dir(model_name, model_dir=None):
    if model_name not in wan_default_config:
        raise ValueError(f"Unsupported model: {model_name}")
    return os.path.expanduser(str(model_dir or os.path.join("wan_models", model_name)))
