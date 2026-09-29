import torch
from inference.prediction import predict_image
from inference.uncertainty import mc_dropout


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(3, 4)
    def forward(self, x): return self.fc(self.drop(x.mean((2, 3))))


def test_prediction_probabilities():
    model = TinyModel()
    result = predict_image(model, torch.ones(1, 3, 8, 8), ["a", "b", "c", "d"], torch.device("cpu"))
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-6
    assert 0 <= result["confidence"] <= 1


def test_mc_dropout_entropy_is_finite_and_restores_eval():
    model = TinyModel().eval()
    result = mc_dropout(model, torch.ones(1, 3, 8, 8), samples=4)
    assert 0 <= result["predictive_entropy"]
    assert model.training is False
    assert model.drop.training is False
