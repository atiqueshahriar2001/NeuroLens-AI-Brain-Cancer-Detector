import pytest
import torch
from models.architectures import build_model
from models.model_loader import load_model


def test_resnet_output_shape():
    model = build_model("ResNet50", 4).eval()
    with torch.inference_mode():
        assert model(torch.zeros(1, 3, 224, 224)).shape == (1, 4)


def test_checkpoint_loads_strictly():
    from config import CHECKPOINT_PATH
    if not CHECKPOINT_PATH.exists(): pytest.skip("Checkpoint not supplied")
    model, classes, name = load_model(CHECKPOINT_PATH, torch.device("cpu"))
    assert name == "ResNet50"
    assert classes == ["glioma", "meningioma", "notumor", "pituitary"]
    assert not model.training
