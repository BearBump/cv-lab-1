#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Контроллер для GUI интерфейса.
Связывает UI с бизнес-логикой.
"""

import cv2
import numpy as np
from typing import Optional
from models.image_processor import ImageProcessor
from models.image_analyzer import ImageAnalyzer
from views.main_window import MainWindow
from controllers.opencv_controller import OpenCVController
from utils.file_utils import validate_image_path


class GUIController:
    """
    Контроллер для управления GUI интерфейсом.
    """
    
    def __init__(self):
        """Инициализация контроллера."""
        self.processor = ImageProcessor()
        self.analyzer = ImageAnalyzer(self.processor)
        self.main_window = MainWindow()
        self.opencv_controller = None
        
        # Устанавливаем callback для обновления
        self.main_window.set_update_callback(self._on_update_requested)
    
    def _on_update_requested(self, *args, **kwargs):
        """
        Обработчик запроса обновления от UI.
        
        Args:
            *args: Аргументы (может быть путь к файлу)
            **kwargs: Ключевые аргументы (может быть show_additional, update_additional)
        """
        # Проверяем, нужно ли загрузить новое изображение
        if args and isinstance(args[0], str):
            self._load_image(args[0])
            return
        
        # Проверяем, нужно ли показать дополнительные окна
        if kwargs.get('show_additional', False):
            self._show_additional_windows()
            return
        
        # Проверяем, нужно ли обновить дополнительные окна
        if kwargs.get('update_additional', False):
            self._update_additional_windows()
            return
        
        # Обычное обновление
        self._update_display()
    
    def _load_image(self, path: str):
        """
        Загружает изображение.
        
        Args:
            path: Путь к изображению
        """
        if not validate_image_path(path):
            self.main_window.update_status(f"Ошибка: Неверный формат файла: {path}")
            return
        
        success = self.processor.load_image(path)
        if success:
            self.main_window.original_image = self.processor.original_image
            self.main_window.update_status(f"Загружено: {path}")
            self._update_display()
        else:
            self.main_window.update_status(f"Ошибка загрузки: {path}")
    
    def _update_display(self):
        """Обновляет отображение."""
        if self.processor.original_image is None:
            return
        
        # Получаем параметры из UI
        ui_params = self.main_window.get_parameters()

        # выбор бэкенда обработки: NumPy или Pillow
        self.processor.use_pillow = (ui_params.get("backend") == "pillow")
        
        # Обновляем параметры процессора
        self.processor.set_parameter("brightness", ui_params["brightness"])
        self.processor.set_parameter("contrast", ui_params["contrast"])
        self.processor.set_parameter("r_offset", ui_params["r_offset"])
        self.processor.set_parameter("g_offset", ui_params["g_offset"])
        self.processor.set_parameter("b_offset", ui_params["b_offset"])
        self.processor.set_parameter("gamma_x10", ui_params["gamma_x10"])
        self.processor.set_parameter("swap_mode", ui_params["swap_mode"])
        self.processor.set_parameter("negate_r", ui_params["negate_r"])
        self.processor.set_parameter("negate_g", ui_params["negate_g"])
        self.processor.set_parameter("negate_b", ui_params["negate_b"])
        self.processor.set_parameter("flip_horizontal", ui_params["flip_horizontal"])
        self.processor.set_parameter("flip_vertical", ui_params["flip_vertical"])
        
        # Обрабатываем изображение
        processed_image = self.processor.process_image()
        self.main_window.processed_image = processed_image
        
        # Обновляем отображение
        self.main_window.update_image_display(processed_image)
        
        # Обновляем zoom окно
        self._update_zoom_display(processed_image)
        
        # Обновляем информацию о пикселе
        self._update_pixel_info(processed_image)
        
        # Обновляем дополнительные окна в правой панели
        self._update_additional_windows()
        
        # Обновляем окно просмотра каналов если оно открыто
        self._update_channel_viewer()
    
    def _update_pixel_info(self, image: np.ndarray):
        """
        Обновляет информацию о текущем пикселе.
        
        Args:
            image: Обработанное изображение
        """
        mouse_x, mouse_y = self.main_window.mouse_pos
        
        # Ограничиваем координаты границами изображения
        height, width = image.shape[:2]
        mouse_x = max(0, min(mouse_x, width - 1))
        mouse_y = max(0, min(mouse_y, height - 1))
        
        # Получаем RGB значения
        blue, green, red = image[mouse_y, mouse_x]
        intensity = self.processor.intensity_at_pixel(image, mouse_x, mouse_y)
        
        # Формируем информацию о пикселе
        pixel_info = f"Позиция: ({mouse_x}, {mouse_y})\n"
        pixel_info += f"RGB: ({int(red)}, {int(green)}, {int(blue)})\n"
        pixel_info += f"Интенсивность: {intensity:.1f}\n"
        pixel_info += f"BGR: ({int(blue)}, {int(green)}, {int(red)})"
        
        self.main_window.update_pixel_info(pixel_info)
        
        # Обновляем статистики окна
        mean_intensity, std_intensity = self.processor.window_mean_std_intensity(
            image, mouse_x, mouse_y
        )
        
        stats_info = f"Среднее: {mean_intensity:.2f}\n"
        stats_info += f"Стд. откл.: {std_intensity:.2f}\n"
        stats_info += f"Окно: 11×11"
        
        self.main_window.update_stats_info(stats_info)
    
    def _update_zoom_display(self, image: np.ndarray):
        """
        Обновляет отображение zoom окна.
        
        Args:
            image: Обработанное изображение
        """
        mouse_x, mouse_y = self.main_window.mouse_pos
        
        # Ограничиваем координаты границами изображения
        height, width = image.shape[:2]
        mouse_x = max(0, min(mouse_x, width - 1))
        mouse_y = max(0, min(mouse_y, height - 1))
        
        # Создаем zoom окно 11x11 с увеличением x8
        zoom_image = self.analyzer.create_zoom_window(image, mouse_x, mouse_y)
        
        # Обновляем отображение
        self.main_window.update_zoom_display(zoom_image)
    
    def _show_additional_windows(self):
        """Показывает дополнительные окна с каналами и гистограммами."""
        if self.processor.original_image is None:
            self.main_window.update_status("Сначала загрузите изображение")
            return
        
        # Обновляем дополнительные окна в правой панели
        self._update_additional_windows()
        self.main_window.update_status("Дополнительные окна обновлены")
    
    def _update_additional_windows(self):
        """Обновляет дополнительные окна в правой панели."""
        if self.processor.original_image is None:
            return
        
        # Создаем окна с каналами и гистограммами
        processed_image = self.processor.process_image()
        channels_mosaic = self.analyzer.create_mosaic_channels(processed_image)
        histograms_image = self.analyzer.make_histogram_image(processed_image)
        
        # Обновляем отображение в правой панели
        self.main_window.update_channels_display(channels_mosaic)
        self.main_window.update_histograms_display(histograms_image)
    
    def _update_channel_viewer(self):
        """Обновляет окно просмотра каналов если оно открыто."""
        if (self.main_window.channel_viewer is not None and 
            self.main_window.channel_viewer.is_visible() and 
            self.processor.processed_image is not None):
            self.main_window.channel_viewer.load_image(self.processor.processed_image)
    
    def run(self):
        """Запускает приложение."""
        self.main_window.run()
