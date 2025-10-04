#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модель для обработки изображений.
Содержит всю бизнес-логику обработки изображений.
"""

import math
from typing import Tuple, Dict, Any
import cv2
import numpy as np
from config.settings import *


class ImageProcessor:
    """`
    Класс для обработки изображений с различными эффектами.
    """
    
    def __init__(self):
        """Инициализация процессора изображений."""
        self.original_image = None
        self.processed_image = None
        self.params_dirty = True
        self.use_pillow = False  # переключатель режима обработки: False=NumPy, True=Pillow

        
        # Параметры обработки
        self.params = {
            "brightness": 100,      # 0..200 (beta = val-100)
            "contrast": 100,        # 50..300 mapped; alpha = val/100
            "r_offset": 100,        # 0..200 (off = val-100)
            "g_offset": 100,
            "b_offset": 100,
            "gamma_x10": 10,        # 5..40  => 0.5..4.0
            "swap_mode": "BGR",     # BGR, BRG, GBR, GRB, RBG, RGB
            "negate_r": False,      # 0/1
            "negate_g": False,
            "negate_b": False,
            "flip_horizontal": False,
            "flip_vertical": False,
        }
    
    def load_image(self, path: str) -> bool:
        """
        Загружает изображение из файла.
        
        Args:
            path: Путь к файлу изображения
            
        Returns:
            True если изображение успешно загружено, False иначе
        """
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return False
            
            if img.ndim == 2:
                # Серое изображение -> в 3 канала
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                # Отбрасываем альфа-канал
                img = img[:, :, :3]
            
            self.original_image = img
            self.processed_image = img.copy()
            self.params_dirty = True
            return True
            
        except Exception:
            return False
    
    def set_parameter(self, param_name: str, value: Any) -> None:
        """
        Устанавливает параметр обработки.
        
        Args:
            param_name: Имя параметра
            value: Значение параметра
        """
        if param_name in self.params:
            self.params[param_name] = value
            self.params_dirty = True
    
    def get_parameter(self, param_name: str) -> Any:
        """
        Получает значение параметра.
        
        Args:
            param_name: Имя параметра
            
        Returns:
            Значение параметра
        """
        return self.params.get(param_name, None)
    
    def process_image(self) -> np.ndarray:
        """
        Обрабатывает изображение согласно текущим параметрам.
        """
        if self.original_image is None or not self.params_dirty:
            return self.processed_image

        processed = self.original_image.copy()

        # Извлекаем параметры из self.params (как и было)
        brightness_offset = self.params["brightness"] - 100
        contrast_factor = self.params["contrast"] / 100.0
        red_offset = self.params["r_offset"] - 100
        green_offset = self.params["g_offset"] - 100
        blue_offset = self.params["b_offset"] - 100
        gamma_value = self.params["gamma_x10"] / 10.0
        swap_mode = self.params["swap_mode"]
        negate_red = self.params["negate_r"]
        negate_green = self.params["negate_g"]
        negate_blue = self.params["negate_b"]
        flip_h = self.params["flip_horizontal"]
        flip_v = self.params["flip_vertical"]

        if getattr(self, "use_pillow", False):
            try:
                # 👇 Берём готовые обёртки из модуля
                from models.pillow_processor import (
                    contrast_pillow,
                    brighten_additive_pillow,
                    channel_offsets_pillow,
                    gamma_pillow,
                    invert_channels_pillow,
                    swap_channels_pillow,
                    flip_pillow,
                )
            except Exception:
                self.use_pillow = False

            if self.use_pillow:
                # Начинаем в BGR-формате (все функции возвращают BGR)
                pil_bgr = processed

                # (1) Контраст
                pil_bgr = contrast_pillow(pil_bgr, contrast_factor)

                # (1a) Яркость — АДДИТИВНО (полное совпадение с NumPy-поведением)
                pil_bgr = brighten_additive_pillow(pil_bgr, brightness_offset)

                # (1b) Сдвиги отдельных каналов
                pil_bgr = channel_offsets_pillow(pil_bgr, red_offset, green_offset, blue_offset)

                # (2) Гамма (через LUT, учитывает settings)
                pil_bgr = gamma_pillow(pil_bgr, gamma_value)

                # (3) Инверсии по выбранным каналам
                pil_bgr = invert_channels_pillow(pil_bgr, negate_red, negate_green, negate_blue)

                # (4) Перестановка каналов
                pil_bgr = swap_channels_pillow(pil_bgr, swap_mode)

                # (5) Отражения
                pil_bgr = flip_pillow(pil_bgr, flip_h, flip_v)

                processed = pil_bgr


        if not getattr(self, "use_pillow", False):
            # --- NumPy РЕЖИМ ---
            processed = self._apply_brightness_contrast_per_channel(
                processed, brightness_offset, red_offset, green_offset, blue_offset, contrast_factor
            )
            processed = self._apply_gamma(processed, gamma_value)
            processed = self._invert_selected_channels(processed, negate_red, negate_green, negate_blue)
            processed = self._swap_channels(processed, swap_mode)
            processed = self._flip_image(processed, flip_h, flip_v)

        self.processed_image = processed
        self.params_dirty = False
        return processed

    
    def to_gray_manual(self, bgr: np.ndarray) -> np.ndarray:
        """
        Конвертирует BGR изображение в серое, используя формулу (R+G+B)/3.
        
        Args:
            bgr: Изображение в формате BGR
            
        Returns:
            Серое изображение в формате uint8
        """
        red_channel = bgr[..., 2].astype(np.float32)
        green_channel = bgr[..., 1].astype(np.float32)
        blue_channel = bgr[..., 0].astype(np.float32)
        
        gray = (red_channel + green_channel + blue_channel) / 3.0
        gray = np.clip(gray, MIN_PIXEL_VALUE, MAX_PIXEL_VALUE).astype(np.uint8)
        
        return gray
    
    def intensity_at_pixel(self, bgr: np.ndarray, x: int, y: int) -> float:
        """
        Вычисляет интенсивность пикселя как среднее значение RGB каналов.
        
        Args:
            bgr: Изображение в формате BGR
            x: Координата X пикселя
            y: Координата Y пикселя
            
        Returns:
            Интенсивность пикселя (0.0 - 255.0)
        """
        blue, green, red = bgr[y, x]
        return (float(red) + float(green) + float(blue)) / 3.0
    
    def window_mean_std_intensity(
        self, bgr: np.ndarray, x: int, y: int, half: int = INNER_WINDOW_HALF
    ) -> Tuple[float, float]:
        """
        Вычисляет среднее и стандартное отклонение интенсивности в окне.
        
        Args:
            bgr: Изображение в формате BGR
            x: Центральная координата X окна
            y: Центральная координата Y окна
            half: Половина размера окна
            
        Returns:
            Кортеж (среднее, стандартное_отклонение)
        """
        height, width, _ = bgr.shape
        
        # Определяем границы окна с учетом границ изображения
        x_start = max(0, x - half)
        y_start = max(0, y - half)
        x_end = min(width, x + half + 1)
        y_end = min(height, y + half + 1)
        
        # Извлекаем окно
        window_patch = bgr[y_start:y_end, x_start:x_end, :]
        
        # Вычисляем интенсивность для каждого пикселя
        red_channel = window_patch[..., 2].astype(np.float64)
        green_channel = window_patch[..., 1].astype(np.float64)
        blue_channel = window_patch[..., 0].astype(np.float64)
        intensity = (red_channel + green_channel + blue_channel) / 3.0
        
        # Вычисляем статистики
        pixel_count = intensity.size # количество пикселей в окне 121 если не у края
        sum_intensity = float(intensity.sum()) # сумма всех интенсивностей в окне
        sum_squared = float((intensity * intensity).sum()) # сумма квадратов интенсивностей. для дисперсии
        
        mean_intensity = sum_intensity / max(pixel_count, 1) # средняя яркость пикселей в окне
        variance = sum_squared / max(pixel_count, 1) - (mean_intensity * mean_intensity) # Дисперсия яркости
        std_deviation = math.sqrt(max(variance, 0.0)) # стандартное отклонение яркости (разброс значений интенсивности)
        
        return mean_intensity, std_deviation
    
    def _apply_brightness_contrast_per_channel(
        self, bgr: np.ndarray, global_beta: int, r_off: int, 
        g_off: int, b_off: int, contrast_alpha: float
    ) -> np.ndarray:
        """Применяет контраст, яркость и сдвиги по каналам."""
        result = bgr.astype(np.float32)
        
        # Применяем контраст и глобальную яркость
        result = (result - 128.0) * contrast_alpha + 128.0 + float(global_beta)
        
        # Применяем смещения по каналам
        result[..., 2] += float(r_off)  # Красный канал
        result[..., 1] += float(g_off)  # Зеленый канал
        result[..., 0] += float(b_off)  # Синий канал
        
        return np.clip(result, MIN_PIXEL_VALUE, MAX_PIXEL_VALUE).astype(np.uint8)
    
    def _apply_gamma(self, bgr: np.ndarray, gamma: float) -> np.ndarray:
        """Применяет гамма-коррекцию к изображению."""
        if abs(gamma - 1.0) < 1e-6:
            return bgr
        
        # Создаем lookup table для гамма-коррекции
        inverse_gamma = 1.0 / max(gamma, 1e-6)
        pixel_values = np.arange(HISTOGRAM_BINS, dtype=np.float32) / MAX_PIXEL_VALUE
        lookup_table = np.clip(
            (pixel_values ** inverse_gamma) * MAX_PIXEL_VALUE, 
            MIN_PIXEL_VALUE, 
            MAX_PIXEL_VALUE
        ).astype(np.uint8)
        
        return cv2.LUT(bgr, lookup_table)
    
    def _invert_selected_channels(
        self, bgr: np.ndarray, neg_r: bool, neg_g: bool, neg_b: bool
    ) -> np.ndarray:
        """Инвертирует выбранные цветовые каналы."""
        result = bgr.copy()
        
        if neg_b:
            result[..., 0] = MAX_PIXEL_VALUE - result[..., 0]  # Синий канал
        if neg_g:
            result[..., 1] = MAX_PIXEL_VALUE - result[..., 1]  # Зеленый канал
        if neg_r:
            result[..., 2] = MAX_PIXEL_VALUE - result[..., 2]  # Красный канал
        
        return result
    
    def _swap_channels(self, bgr: np.ndarray, mode) -> np.ndarray:
        """Переставляет цветовые каналы согласно заданному режиму."""
        blue_channel, green_channel, red_channel = cv2.split(bgr)
        
        # Поддержка как строковых, так и числовых значений
        if isinstance(mode, str):
            channel_combinations = {
                "BGR": [blue_channel, green_channel, red_channel],
                "BRG": [blue_channel, red_channel, green_channel],
                "GBR": [green_channel, blue_channel, red_channel],
                "GRB": [green_channel, red_channel, blue_channel],
                "RBG": [red_channel, blue_channel, green_channel],
                "RGB": [red_channel, green_channel, blue_channel],
            }
        else:
            # Обратная совместимость с числовыми значениями
            channel_combinations = {
                0: [blue_channel, green_channel, red_channel],   # BGR
                1: [blue_channel, red_channel, green_channel],   # BRG
                2: [green_channel, blue_channel, red_channel],   # GBR
                3: [green_channel, red_channel, blue_channel],   # GRB
                4: [red_channel, blue_channel, green_channel],   # RBG
                5: [red_channel, green_channel, blue_channel],   # RGB
            }
        
        if mode in channel_combinations:
            return cv2.merge(channel_combinations[mode])
        
        return bgr
    
    def _flip_image(self, bgr: np.ndarray, flip_h: bool, flip_v: bool) -> np.ndarray:
        """Применяет отражения изображения по горизонтали и/или вертикали."""
        result = bgr
        
        if flip_h and flip_v:
            # Отражение по обеим осям
            result = cv2.flip(result, -1)
        elif flip_h:
            # Отражение по горизонтали
            result = cv2.flip(result, 1)
        elif flip_v:
            # Отражение по вертикали
            result = cv2.flip(result, 0)
        
        return result
