"""MRI image validation and model preprocessing."""
from io import BytesIO
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torchvision import transforms
from config import IMAGE_SIZE

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_IMAGE_PIXELS = 40_000_000
PREPROCESS = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_image(data: bytes) -> Image.Image:
    if not data:
        raise ValueError("The uploaded file is empty.")
    try:
        image = Image.open(BytesIO(data))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The uploaded file is not a valid, readable image.") from exc
    if image.width <= 0 or image.height <= 0 or image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("The image dimensions are too large to process safely.")
    try:
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The uploaded file is not a valid, readable image.") from exc


def image_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    return PREPROCESS(image).unsqueeze(0).to(device)
