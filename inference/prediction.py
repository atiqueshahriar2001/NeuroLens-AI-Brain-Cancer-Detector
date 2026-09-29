"""Prediction and probability validation."""
import torch


def predict_image(model, image, class_names, device):
    model.eval()
    with torch.inference_mode():
        logits = model(image.to(device))
        if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != len(class_names):
            raise ValueError(f"Invalid model output shape: {tuple(logits.shape)}")
        probs = torch.softmax(logits, dim=1)[0]
        if not torch.isfinite(probs).all():
            raise ValueError("Model returned non-finite probabilities.")
        values = probs.cpu().tolist()
        if any(p < 0 or p > 1 for p in values) or abs(sum(values) - 1) > 1e-5:
            raise ValueError("Model returned invalid probabilities.")
    index = int(probs.argmax().item())
    return {"class_name": class_names[index], "class_index": index, "confidence": values[index], "probabilities": dict(zip(class_names, values))}
