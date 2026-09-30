"""Application configuration and checkpoint metadata."""
from pathlib import Path
import os
import secrets
import torch
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "neurolens_best.pth"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = 224
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
try:
    MC_SAMPLES = int(os.getenv("MC_SAMPLES", "8"))
except ValueError as exc:
    raise RuntimeError("MC_SAMPLES must be an integer greater than or equal to 2.") from exc
if MC_SAMPLES < 2:
    raise RuntimeError("MC_SAMPLES must be greater than or equal to 2.")
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_hex(32)
    if os.getenv("FLASK_DEBUG", "0").lower() not in {"1", "true"}:
        import logging
        logging.getLogger(__name__).warning("SECRET_KEY is unset; sessions will be invalidated whenever the process restarts.")
HISTORY_DB_PATH = Path(os.getenv("HISTORY_DB_PATH", "data/history.sqlite3"))
if not HISTORY_DB_PATH.is_absolute():
    HISTORY_DB_PATH = BASE_DIR / HISTORY_DB_PATH
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
