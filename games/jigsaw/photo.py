from __future__ import annotations
import cv2
import numpy as np
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
def ask_file() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="pick a picture",
        filetypes=[("pictures", " ".join("*" + e for e in EXTS)),
                   ("all files", "*.*")])
    root.destroy()
    return path or None
def load(path: str) -> np.ndarray | None:
    try:
        raw = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if raw.size == 0:
        return None
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img
def fit(image: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    s = min(max_w / w, max_h / h)
    nw = max(1, int(round(w * s)))
    nh = max(1, int(round(h * s)))
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (nw, nh), interpolation=interp)
def snap_to_grid(image: np.ndarray, rows: int, cols: int) -> np.ndarray:
    h, w = image.shape[:2]
    nw = (w // cols) * cols
    nh = (h // rows) * rows
    if nw == w and nh == h:
        return image
    x = (w - nw) // 2
    y = (h - nh) // 2
    return image[y:y + nh, x:x + nw].copy()
