"""Image serialization helpers for Flask templates and session records."""
import base64
from io import BytesIO
from PIL import Image


def image_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
