"""Integrated Gradients using Captum when installed."""
import numpy as np
import torch


def integrated_gradients(model, image, target_class, n_steps=16):
    try:
        from captum.attr import IntegratedGradients
    except ImportError as exc:
        raise RuntimeError("Integrated Gradients unavailable: install captum.") from exc
    if n_steps < 1: raise ValueError("n_steps must be positive.")
    model.eval()
    baseline = torch.zeros_like(image, device=image.device)
    attribution = IntegratedGradients(model).attribute(
        image,
        baselines=baseline,
        target=target_class,
        n_steps=n_steps,
        internal_batch_size=1,
    )
    result = attribution.abs().sum(1)[0]
    result = torch.nan_to_num(result)
    if result.max() > 0: result = result / result.max()
    return result.detach().cpu().numpy()
