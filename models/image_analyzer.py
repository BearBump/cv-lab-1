#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модель для анализа изображений.
Содержит функции для создания гистограмм, мозаик и других аналитических представлений.
"""

import cv2
import numpy as np
from typing import Tuple
from config.settings import *


class ImageAnalyzer:
    """
    Класс для анализа изображений и создания визуальных представлений.
    """
    
    def __init__(self, processor):
        """
        Инициализация анализатора.
        
        Args:
            processor: Экземпляр ImageProcessor
        """
        self.processor = processor
    
    def make_histogram_image(self, bgr: np.ndarray) -> np.ndarray:
        """
        Создает изображение с отдельными гистограммами для каждого канала.
        
        Args:
            bgr: Изображение в формате BGR
            
        Returns:
            Изображение с гистограммами размером HISTOGRAM_HEIGHT x HISTOGRAM_WIDTH
        """
        canvas = np.zeros((HISTOGRAM_HEIGHT, HISTOGRAM_WIDTH, 3), dtype=np.uint8)

        # Вычисляем гистограммы
        gray_image = self.processor.to_gray_manual(bgr)
        gray_histogram = np.bincount(gray_image.ravel(), minlength=HISTOGRAM_BINS).astype(np.float32)

        # Гистограммы для каждого канала
        red_channel = bgr[..., 2]
        green_channel = bgr[..., 1]
        blue_channel = bgr[..., 0]
        
        red_histogram = np.bincount(red_channel.ravel(), minlength=HISTOGRAM_BINS).astype(np.float32)
        green_histogram = np.bincount(green_channel.ravel(), minlength=HISTOGRAM_BINS).astype(np.float32)
        blue_histogram = np.bincount(blue_channel.ravel(), minlength=HISTOGRAM_BINS).astype(np.float32)

        # Размеры для каждой отдельной гистограммы (2x2 сетка)
        hist_width = HISTOGRAM_WIDTH // 2
        hist_height = HISTOGRAM_HEIGHT // 2
        
        def normalize_histogram(histogram: np.ndarray, target_height: int) -> np.ndarray:
            """Нормирует гистограмму на заданную высоту."""
            if histogram.max() < 1e-9:
                return histogram
            return (histogram / histogram.max()) * (target_height - 20)

        def draw_single_histogram(histogram: np.ndarray, color: Tuple[int, int, int], 
                                x_offset: int, y_offset: int, label: str) -> None:
            """Рисует отдельную гистограмму в указанной области."""
            # Нормируем гистограмму для данной области
            normalized_hist = normalize_histogram(histogram, hist_height)
            
            # Рисуем рамку для этой гистограммы
            cv2.rectangle(canvas, 
                         (x_offset, y_offset), 
                         (x_offset + hist_width - 1, y_offset + hist_height - 1), 
                         COLOR_GRAY, 1)
            
            # Рисуем гистограмму
            points = []
            for i in range(HISTOGRAM_BINS):
                x_coord = x_offset + int(i * (hist_width - 1) / (HISTOGRAM_BINS - 1))
                y_coord = y_offset + hist_height - 10 - int(normalized_hist[i])
                points.append((x_coord, y_coord))
            
            for i in range(1, len(points)):
                cv2.line(canvas, points[i - 1], points[i], color, 1, cv2.LINE_AA)
            
            # Добавляем подпись
            cv2.putText(
                canvas, 
                label, 
                (x_offset + 5, y_offset + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                1, 
                cv2.LINE_AA
            )

        # Рисуем 4 отдельные гистограммы в сетке 2x2
        # Верхний ряд: Gray (слева), Red (справа)
        draw_single_histogram(gray_histogram, COLOR_WHITE, 0, 0, "Gray")
        draw_single_histogram(red_histogram, COLOR_RED, hist_width, 0, "Red")
        
        # Нижний ряд: Green (слева), Blue (справа)
        draw_single_histogram(green_histogram, COLOR_GREEN, 0, hist_height, "Green")
        draw_single_histogram(blue_histogram, COLOR_BLUE, hist_width, hist_height, "Blue")
        
        return canvas
    
    def create_mosaic_channels(self, bgr: np.ndarray) -> np.ndarray:
        """
        Создает мозаику 2x2 с каналами изображения.
        
        Расположение:
            Gray | R
            G    | B
        
        Args:
            bgr: Изображение в формате BGR
            
        Returns:
            Мозаика размером 2*MOSAIC_TILE_SIZE x 2*MOSAIC_TILE_SIZE
        """
        # Извлекаем каналы
        gray_image = self.processor.to_gray_manual(bgr)
        red_channel = bgr[..., 2]
        green_channel = bgr[..., 1]
        blue_channel = bgr[..., 0]

        def scale_image_to_tile(image: np.ndarray) -> np.ndarray:
            """Масштабирует изображение до размера тайла."""
            height, width = image.shape[:2]
            scale_factor = MOSAIC_TILE_SIZE / max(height, width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        def convert_to_3_channels(image: np.ndarray) -> np.ndarray:
            """Конвертирует одноканальное изображение в трехканальное."""
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return image

        # Создаем тайлы
        tiles = [
            convert_to_3_channels(scale_image_to_tile(gray_image)),   # Gray
            convert_to_3_channels(scale_image_to_tile(red_channel)),  # R
            convert_to_3_channels(scale_image_to_tile(green_channel)), # G
            convert_to_3_channels(scale_image_to_tile(blue_channel))  # B
        ]

        # Создаем полотно для мозаики
        canvas_size = 2 * MOSAIC_TILE_SIZE
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

        def place_tile(image: np.ndarray, row: int, col: int) -> None:
            """Размещает тайл на полотне мозаики."""
            height, width = image.shape[:2]
            y_start = row * MOSAIC_TILE_SIZE
            x_start = col * MOSAIC_TILE_SIZE
            canvas[y_start:y_start + height, x_start:x_start + width] = image

        # Размещаем тайлы: Gray(0,0), R(0,1), G(1,0), B(1,1)
        place_tile(tiles[0], 0, 0)  # Gray
        place_tile(tiles[1], 0, 1)  # R
        place_tile(tiles[2], 1, 0)  # G
        place_tile(tiles[3], 1, 1)  # B

        # Добавляем подписи
        label_positions = [
            ("Gray", (8, 20), COLOR_TEXT),
            ("R", (MOSAIC_TILE_SIZE + 8, 20), COLOR_RED),
            ("G", (8, MOSAIC_TILE_SIZE + 20), COLOR_GREEN),
            ("B", (MOSAIC_TILE_SIZE + 8, MOSAIC_TILE_SIZE + 20), COLOR_BLUE)
        ]
        
        for label, position, color in label_positions:
            cv2.putText(
                canvas, 
                label, 
                position, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                color, 
                1, 
                cv2.LINE_AA
            )

        return canvas
    
    def create_zoom_window(self, processed_image: np.ndarray, mouse_x: int, mouse_y: int) -> np.ndarray:
        """
        Создает увеличенное окно 11x11 вокруг позиции мыши.
        
        Args:
            processed_image: Обработанное изображение
            mouse_x: Координата X мыши
            mouse_y: Координата Y мыши
            
        Returns:
            Увеличенный фрагмент изображения
        """
        height, width = processed_image.shape[:2]
        
        # Определяем границы окна 11x11
        x_start = max(0, mouse_x - INNER_WINDOW_HALF)
        y_start = max(0, mouse_y - INNER_WINDOW_HALF)
        x_end = min(width, mouse_x + INNER_WINDOW_HALF + 1)
        y_end = min(height, mouse_y + INNER_WINDOW_HALF + 1)
        
        # Извлекаем фрагмент
        patch = processed_image[y_start:y_end, x_start:x_end]
        
        # Увеличиваем в 8 раз
        zoomed_patch = cv2.resize(
            patch, 
            (patch.shape[1] * ZOOM_FACTOR, patch.shape[0] * ZOOM_FACTOR), 
            interpolation=cv2.INTER_NEAREST
        )
        
        return zoomed_patch
    
    def draw_frame_and_status(
        self, image: np.ndarray, mouse_x: int, mouse_y: int, processed_image: np.ndarray
    ) -> np.ndarray:
        """
        Рисует рамку вокруг курсора и статусную информацию.
        
        Args:
            image: Изображение для рисования
            mouse_x: Координата X мыши
            mouse_y: Координата Y мыши
            processed_image: Обработанное изображение для вычислений
            
        Returns:
            Изображение с рамкой и статусом
        """
        result_image = image.copy()
        height, width = result_image.shape[:2]
        
        # Рисуем внешнюю рамку 13x13
        x_start = max(0, mouse_x - OUTER_FRAME_HALF)
        y_start = max(0, mouse_y - OUTER_FRAME_HALF)
        x_end = min(width - 1, mouse_x + OUTER_FRAME_HALF)
        y_end = min(height - 1, mouse_y + OUTER_FRAME_HALF)
        
        cv2.rectangle(result_image, (x_start, y_start), (x_end, y_end), COLOR_YELLOW, 1, cv2.LINE_8)
        
        # Вычисляем статистики для окна 11x11
        intensity = self.processor.intensity_at_pixel(processed_image, mouse_x, mouse_y)
        mean_intensity, std_intensity = self.processor.window_mean_std_intensity(processed_image, mouse_x, mouse_y)
        
        # Получаем RGB значения
        blue, green, red = processed_image[mouse_y, mouse_x]
        
        # Формируем текст статуса
        status_text = (
            f"p=({mouse_x},{mouse_y})  "
            f"RGB=({int(red)}, {int(green)}, {int(blue)})  "
            f"I={intensity:.1f}  "
            f"W_mean={mean_intensity:.2f}  "
            f"W_std={std_intensity:.2f}"
        )
        
        # Рисуем текст с тенью для лучшей читаемости
        cv2.putText(
            result_image, 
            status_text, 
            (10, 24), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.65, 
            COLOR_BLACK, 
            3, 
            cv2.LINE_AA
        )
        cv2.putText(
            result_image, 
            status_text, 
            (10, 24), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.65, 
            COLOR_WHITE, 
            1, 
            cv2.LINE_AA
        )
        
        return result_image
