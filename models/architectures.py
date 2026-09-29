"""Small model factory for the supported torchvision architectures."""
import torch.nn as nn
from torchvision import models


def build_model(name: str, num_classes: int):
    key = name.lower().replace("_", "").replace("-", "")
    if key == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(nn.Linear(2048, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, num_classes))
    elif key in {"efficientnetb0", "mobilenetv3small"}:
        if key == "efficientnetb0":
            model = models.efficientnet_b0(weights=None)
            model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.classifier[1].in_features, num_classes))
        else:
            model = models.mobilenet_v3_small(weights=None)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif key == "customcnn":
        model = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, num_classes))
    else:
        raise ValueError(f"Unsupported architecture: {name}")
    return model


def get_target_layer(model, model_name: str):
    key = model_name.lower().replace("_", "").replace("-", "")
    if key == "resnet50":
        return model.layer4[-1]
    if key == "efficientnetb0":
        return model.features[-1]
    if key == "mobilenetv3small":
        return model.features[-1]
    if key == "customcnn":
        return model[3]
    raise ValueError(f"Unsupported architecture: {model_name}")
