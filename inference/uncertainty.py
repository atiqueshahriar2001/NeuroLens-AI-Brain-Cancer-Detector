"""Monte Carlo dropout estimates with batch normalization kept in eval mode."""
import math
import torch
from torch import nn


def mc_dropout(model, image, samples=4):
    if samples < 2:
        raise ValueError("MC Dropout requires at least two samples.")
    model.eval()
    dropout_types = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout)
    changed = []
    for module in model.modules():
        if isinstance(module, dropout_types):
            changed.append((module, module.training))
            module.train()
    try:
        with torch.no_grad():
            probabilities = torch.stack([torch.softmax(model(image), dim=1)[0] for _ in range(samples)])
    finally:
        for module, was_training in changed:
            module.train(was_training)
        model.eval()
    mean = probabilities.mean(dim=0)
    std = probabilities.std(dim=0, unbiased=False)
    entropy = -(mean * mean.clamp_min(1e-12).log()).sum().item()
    return {"mean_probabilities": mean.cpu().tolist(), "std_probabilities": std.cpu().tolist(), "predictive_entropy": entropy, "prediction_uncertainty": entropy / math.log(mean.numel()) if mean.numel() > 1 else 0.0}
