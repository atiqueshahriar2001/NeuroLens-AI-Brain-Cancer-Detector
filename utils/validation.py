"""Upload filename and extension checks."""
from pathlib import Path
from werkzeug.utils import secure_filename
from inference.preprocessing import ALLOWED_EXTENSIONS


def validate_filename(filename: str) -> str:
    safe = secure_filename(filename or "")
    if not safe or Path(safe).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported image type. Use JPG, JPEG, PNG, BMP, or WEBP.")
    return safe
