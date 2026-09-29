"""Strict loading of checkpoint metadata and model weights."""
import logging
from pathlib import Path
import torch
from models.architectures import build_model

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"Could not read checkpoint safely: {exc}") from exc
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain model_state_dict metadata.")
    name = checkpoint.get("best_model_name")
    classes = checkpoint.get("class_names")
    count = checkpoint.get("num_classes")
    if not name or not isinstance(classes, (list, tuple)) or not isinstance(count, int):
        raise ValueError("Checkpoint must contain best_model_name, class_names, and num_classes.")
    if len(classes) != count:
        raise ValueError(f"Checkpoint class metadata mismatch: {len(classes)} names for {count} outputs.")
    state = checkpoint["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint model_state_dict is invalid.")
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model = build_model(name, count)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Checkpoint architecture mismatch for {name}: {exc}") from exc
    model.to(device).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 224, 224, device=device))
    if output.ndim != 2 or output.shape != (1, count):
        raise RuntimeError(f"Model output shape mismatch: expected (1, {count}), got {tuple(output.shape)}")
    return model, list(classes), name
