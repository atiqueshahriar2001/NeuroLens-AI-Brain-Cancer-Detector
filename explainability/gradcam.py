"""Grad-CAM++ attribution and heatmap overlay."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def generate_gradcam_pp(model, model_name, image, target_class):
    # The notebook targets the final convolutional layer across architectures.
    layer = next((module for module in reversed(list(model.modules())) if isinstance(module, nn.Conv2d)), None)
    if layer is None:
        raise ValueError("No convolutional layer found for Grad-CAM++.")
    activation, gradient = [], []
    h1 = layer.register_forward_hook(lambda _m, _i, out: activation.append(out))
    h2 = layer.register_full_backward_hook(lambda _m, _gi, go: gradient.append(go[0]))
    try:
        model.eval(); model.zero_grad(set_to_none=True)
        output = model(image)
        if output.ndim != 2 or not 0 <= target_class < output.shape[1]:
            raise ValueError("Invalid Grad-CAM target class.")
        output[0, target_class].backward()
        if not activation or not gradient or activation[0].ndim != 4 or gradient[0].ndim != 4:
            raise RuntimeError("Grad-CAM target layer did not produce valid 4D activations and gradients.")
        a, g = activation[0], gradient[0]
        grads2, grads3 = g.square(), g.pow(3)
        denominator = 2 * grads2 + (a * grads3).sum((2, 3), keepdim=True)
        alpha = grads2 / torch.where(denominator.abs() > 1e-12, denominator, torch.ones_like(denominator))
        alpha = torch.where(denominator.abs() > 1e-12, alpha, torch.zeros_like(alpha))
        weights = (alpha * F.relu(g)).sum((2, 3), keepdim=True)
        cam = F.relu((weights * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = torch.nan_to_num(cam)
        if cam.max() > 0: cam = cam / cam.max()
        return cam.detach().cpu().numpy()
    finally:
        h1.remove(); h2.remove(); model.zero_grad(set_to_none=True)


def overlay_heatmap(image, cam):
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    heat = np.nan_to_num(np.asarray(cam, dtype=np.float32))
    if heat.ndim != 2: raise ValueError("CAM must be a 2D array.")
    heat = np.clip(heat, 0, 1)
    heat_image = Image.fromarray(np.uint8(heat * 255)).resize(image.size, Image.Resampling.BILINEAR)
    colored = plt.get_cmap("jet")(np.asarray(heat_image) / 255.0)[..., :3]
    return Image.fromarray(np.uint8(np.clip(0.55 * base + 0.45 * colored, 0, 1) * 255))
