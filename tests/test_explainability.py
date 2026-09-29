import numpy as np
import torch
from PIL import Image
from explainability.eas import compute_eas
from explainability.gradcam import overlay_heatmap
from explainability.integrated_gradients import integrated_gradients


def test_eas_bounds_and_heatmap_resize():
    assert 0 <= compute_eas(np.zeros((7, 7)), np.zeros((14, 14))) <= 1
    output = overlay_heatmap(Image.new("RGB", (80, 60)), np.ones((7, 7)))
    assert output.size == (80, 60)


def test_integrated_gradients_small_model():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2))
    values = integrated_gradients(model, torch.ones(1, 3, 2, 2), 0, n_steps=3)
    assert values.shape == (2, 2)
    assert np.isfinite(values).all()
