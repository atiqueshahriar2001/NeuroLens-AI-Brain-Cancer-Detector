import pytest
from PIL import Image
from inference.preprocessing import load_image
from io import BytesIO


def test_grayscale_is_converted_to_rgb():
    buffer = BytesIO(); Image.new("L", (12, 9)).save(buffer, format="PNG")
    image = load_image(buffer.getvalue())
    assert image.mode == "RGB"


def test_invalid_image_is_rejected():
    with pytest.raises(ValueError): load_image(b"broken")


def test_extreme_dimensions_are_rejected_before_decode(monkeypatch):
    class HugeImage:
        width, height = 100_000, 100_000
    monkeypatch.setattr("inference.preprocessing.Image.open", lambda *_args, **_kwargs: HugeImage())
    with pytest.raises(ValueError, match="dimensions"):
        load_image(b"image header")
