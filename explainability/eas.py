"""EAS compares two explanation masks; it is not clinical validation."""
import numpy as np
from PIL import Image


def compute_eas(cam, ig):
    def resize_map(value):
        arr = np.nan_to_num(np.array(value, dtype=np.float32, copy=True))
        if arr.ndim != 2: raise ValueError("Explanation maps must be 2D.")
        arr = np.clip(arr, 0, None)
        if arr.max() > 0: arr /= arr.max()
        image = Image.fromarray(np.uint8(arr * 255)).resize((14, 14), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255
    a, b = resize_map(cam), resize_map(ig)
    a = (a > 0) & (a >= np.percentile(a, 75))
    b = (b > 0) & (b >= np.percentile(b, 75))
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0
