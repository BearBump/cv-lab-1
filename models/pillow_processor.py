from typing import Tuple
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

# --- Конвертеры между OpenCV BGR <-> Pillow RGB ---

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    # BGR -> RGB
    rgb = bgr[..., ::-1].copy()
    return Image.fromarray(rgb, mode="RGB")

def pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"), dtype=np.uint8)
    # RGB -> BGR
    return rgb[..., ::-1].copy()

# --- A) Яркость (глобальная) и смещения каналов ---

def brighten_pillow(bgr: np.ndarray, factor: float) -> np.ndarray:
    """factor=1.0 без изменения; >1 светлее; <1 темнее"""
    pil = bgr_to_pil(bgr)
    pil = ImageEnhance.Brightness(pil).enhance(max(factor, 0.0))
    return pil_to_bgr(pil)

def channel_offsets_pillow(bgr: np.ndarray, r_off: int, g_off: int, b_off: int) -> np.ndarray:
    pil = bgr_to_pil(bgr).convert("RGB")
    r, g, b = pil.split()
    clamp = lambda v: max(0, min(255, v))
    r = r.point(lambda v: clamp(int(v) + int(r_off)))
    g = g.point(lambda v: clamp(int(v) + int(g_off)))
    b = b.point(lambda v: clamp(int(v) + int(b_off)))
    return pil_to_bgr(Image.merge("RGB", (r, g, b)))

# --- B) Контраст ---

def contrast_pillow(bgr: np.ndarray, factor: float) -> np.ndarray:
    """factor=1.0 без изменения; >1 контраст ↑; <1 контраст ↓"""
    pil = bgr_to_pil(bgr)
    pil = ImageEnhance.Contrast(pil).enhance(max(factor, 0.0))
    return pil_to_bgr(pil)

# --- C) Негатив (по каналам) ---

def invert_channels_pillow(bgr: np.ndarray, neg_r: bool, neg_g: bool, neg_b: bool) -> np.ndarray:
    pil = bgr_to_pil(bgr).convert("RGB")
    r, g, b = pil.split()
    if neg_r: r = ImageOps.invert(r)
    if neg_g: g = ImageOps.invert(g)
    if neg_b: b = ImageOps.invert(b)
    return pil_to_bgr(Image.merge("RGB", (r, g, b)))

# --- D) Перестановка каналов ---

def swap_channels_pillow(bgr: np.ndarray, mode: str) -> np.ndarray:
    pil = bgr_to_pil(bgr).convert("RGB")
    r, g, b = pil.split()
    mapping = {
        "RGB": (r, g, b),
        "RBG": (r, b, g),
        "GRB": (g, r, b),
        "GBR": (g, b, r),
        "BRG": (b, r, g),
        "BGR": (b, g, r),
    }
    out = Image.merge("RGB", mapping.get(mode, (r, g, b)))
    return pil_to_bgr(out)

# --- E) Отражения ---

def flip_pillow(bgr: np.ndarray, horiz: bool, vert: bool) -> np.ndarray:
    pil = bgr_to_pil(bgr)
    if horiz:
        pil = ImageOps.mirror(pil)
    if vert:
        pil = ImageOps.flip(pil)
    return pil_to_bgr(pil)

# --- F) Гамма (собственная через LUT) ---

def gamma_pillow(bgr: np.ndarray, gamma: float) -> np.ndarray:
    pil = bgr_to_pil(bgr).convert("RGB")
    g = max(gamma, 1e-6)
    inv = 1.0 / g
    lut = [int(((i / 255.0) ** inv) * 255 + 0.5) for i in range(256)]
    return pil_to_bgr(pil.point(lut * 3))
