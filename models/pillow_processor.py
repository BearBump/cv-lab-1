from typing import Tuple
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

# ВАЖНО: тянем диапазоны и константы из настроек проекта,
# чтобы поведение Pillow-ветки совпадало с NumPy-веткой.
from config.settings import (
    MIN_PIXEL_VALUE,
    MAX_PIXEL_VALUE,
    HISTOGRAM_BINS,
)

# --- Вспомогательное: clamp согласно settings ---
def _clamp(v: int) -> int:
    return max(MIN_PIXEL_VALUE, min(MAX_PIXEL_VALUE, v))

# --- Конвертеры между OpenCV BGR <-> Pillow RGB ---
def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    # BGR -> RGB
    rgb = bgr[..., ::-1].copy()
    return Image.fromarray(rgb, mode="RGB")

def pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"), dtype=np.uint8)
    # RGB -> BGR
    return rgb[..., ::-1].copy()

# ---------------------------------------------------------------------
# A) ЯРКОСТЬ
# ---------------------------------------------------------------------
# 1) Аддитивная яркость (полное совпадение с твоей ручной математикой):
#    value' = value + beta, где beta = brightness - 100
def brighten_additive_pillow(bgr: np.ndarray, beta: int) -> np.ndarray:
    pil = bgr_to_pil(bgr).convert("RGB")
    r, g, b = pil.split()
    r = r.point(lambda v: _clamp(int(v) + int(beta)))
    g = g.point(lambda v: _clamp(int(v) + int(beta)))
    b = b.point(lambda v: _clamp(int(v) + int(beta)))
    return pil_to_bgr(Image.merge("RGB", (r, g, b)))

# 2) Мультипликативная яркость (классический Pillow-вариант — оставляем как опцию)
#    ВНИМАНИЕ: factor=0 делает картинку чёрной (это и отличает его от NumPy-аддитивного)
def brighten_multiplicative_pillow(bgr: np.ndarray, factor: float) -> np.ndarray:
    pil = bgr_to_pil(bgr)
    factor = max(0.0, float(factor))
    pil = ImageEnhance.Brightness(pil).enhance(factor)
    return pil_to_bgr(pil)

# Сдвиги отдельных каналов (R/G/B) с clamp из settings
def channel_offsets_pillow(bgr: np.ndarray, r_off: int, g_off: int, b_off: int) -> np.ndarray:
    pil = bgr_to_pil(bgr).convert("RGB")
    r, g, b = pil.split()
    r = r.point(lambda v: _clamp(int(v) + int(r_off)))
    g = g.point(lambda v: _clamp(int(v) + int(g_off)))
    b = b.point(lambda v: _clamp(int(v) + int(b_off)))
    return pil_to_bgr(Image.merge("RGB", (r, g, b)))

# ---------------------------------------------------------------------
# B) КОНТРАСТ
# ---------------------------------------------------------------------
# Остаётся через ImageEnhance.Contrast, это линейное растяжение;
# factor=1.0 — без изменений.
def contrast_pillow(bgr: np.ndarray, factor: float) -> np.ndarray:
    pil = bgr_to_pil(bgr)
    factor = max(0.0, float(factor))
    pil = ImageEnhance.Contrast(pil).enhance(factor)
    return pil_to_bgr(pil)

# ---------------------------------------------------------------------
# C) НЕГАТИВ ПО КАНАЛАМ
# ---------------------------------------------------------------------
def invert_channels_pillow(bgr: np.ndarray, neg_r: bool, neg_g: bool, neg_b: bool) -> np.ndarray:
    pil = bgr_to_pil(bgr).convert("RGB")
    r, g, b = pil.split()
    if neg_r: r = ImageOps.invert(r)
    if neg_g: g = ImageOps.invert(g)
    if neg_b: b = ImageOps.invert(b)
    return pil_to_bgr(Image.merge("RGB", (r, g, b)))

# ---------------------------------------------------------------------
# D) ПЕРЕСТАНОВКА КАНАЛОВ
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# E) ОТРАЖЕНИЯ
# ---------------------------------------------------------------------
def flip_pillow(bgr: np.ndarray, horiz: bool, vert: bool) -> np.ndarray:
    pil = bgr_to_pil(bgr)
    if horiz:
        pil = ImageOps.mirror(pil)
    if vert:
        pil = ImageOps.flip(pil)
    return pil_to_bgr(pil)

# ---------------------------------------------------------------------
# F) ГАММА (через LUT), c использованием HISTOGRAM_BINS/MAX_PIXEL_VALUE
# ---------------------------------------------------------------------
def gamma_pillow(bgr: np.ndarray, gamma: float) -> np.ndarray:
    pil = bgr_to_pil(bgr).convert("RGB")
    g = max(float(gamma), 1e-6)
    inv = 1.0 / g

    # Строим LUT в соответствии с твоими настройками (0..MAX_PIXEL_VALUE)
    pixel_values = np.arange(HISTOGRAM_BINS, dtype=np.float32) / float(MAX_PIXEL_VALUE)
    lut = np.clip(
        (pixel_values ** inv) * float(MAX_PIXEL_VALUE),
        float(MIN_PIXEL_VALUE),
        float(MAX_PIXEL_VALUE),
    ).astype(np.uint8).tolist()

    return pil_to_bgr(pil.point(lut * 3))
