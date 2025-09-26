#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Конфигурационные настройки приложения.
"""

# Размеры окон и рамок
OUTER_FRAME_HALF = 6  # 13x13 рамка
INNER_WINDOW_HALF = 5  # 11x11 окно для анализа
ZOOM_FACTOR = 8  # Коэффициент увеличения для окна zoom

# Размеры для отображения
HISTOGRAM_HEIGHT = 300
HISTOGRAM_WIDTH = 512
MOSAIC_TILE_SIZE = 256

# Диапазоны значений
MAX_PIXEL_VALUE = 255
MIN_PIXEL_VALUE = 0
HISTOGRAM_BINS = 256

# Цвета для отображения
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_GRAY = (80, 80, 80)
COLOR_LIGHT_GRAY = (200, 200, 200)
COLOR_TEXT = (220, 220, 220)

# Параметры трекбаров
TRACKBAR_RANGES = {
    "Bright": (0, 200),
    "Contrast": (50, 300),
    "R_off": (0, 200),
    "G_off": (0, 200),
    "B_off": (0, 200),
    "Gamma_x10": (5, 40),
    "SwapMode": (0, 5),
    "NegR": (0, 1),
    "NegG": (0, 1),
    "NegB": (0, 1),
}

# Названия окон
WINDOW_NAMES = {
    "IMAGE": "Image",
    "ZOOM": "Zoom 11x11 x8",
    "CONTROLS": "Controls",
    "CHANNELS": "Channels",
    "HISTOGRAMS": "Histograms",
}

# Настройки GUI
GUI_SETTINGS = {
    "window_title": "Лаба 1",
    "min_window_size": (800, 600),
    "default_image_size": (640, 480),
    "update_interval": 15,  # мс
}

# Поддерживаемые форматы изображений
SUPPORTED_FORMATS = ['.bmp', '.png', '.tiff', '.tif', '.jpg', '.jpeg']
