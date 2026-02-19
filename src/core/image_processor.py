from pathlib import Path
from typing import Callable

import numpy as np

import config
from core.artwork import Artwork
from logger import log


class ImageProcessor:
    def __init__(
        self,
        artwork: Artwork,
        save_path: Path,
    ):
        self.artwork = artwork
        self.save_path = save_path

    def _apply_filter(self, name: str, options: dict):
        """Унифицированное применение фильтров и сохранение"""
        log.info("Сравнение %s...", name)
        handmade = options["handmade"]()
        opencv2 = options["opencv"]()
        self.artwork.save_image(
            handmade,
            self.save_path / options["handmade_path"],
        )
        self.artwork.save_image(
            opencv2,
            self.save_path / options["opencv_path"],
        )

    def process_artwork(
        self,
        gamma_param: float = config.GAMMA_CORRECTION_PARAM,
        kernel_size: int = config.KERNEL_GAUSSIAN_SIZE,
    ):
        tasks = {
            "grayscale": {
                "handmade": self.artwork.handmade_grayscale,
                "opencv": self.artwork.opencv_grayscale,
                "handmade_path": "gray_handmade.jpg",
                "opencv_path": "gray_opencv.jpg",
            },
            "gaussian blur": {
                "handmade": lambda : self.artwork.handmade_gaussian_blur(kernel_size=kernel_size),
                "opencv": lambda : self.artwork.opencv_gaussian_blur(kernel_size=kernel_size),
                "handmade_path": "blur_handmade.jpg",
                "opencv_path": "blur_opencv.jpg",
            },
            "edges": {
                "handmade": self.artwork.handmade_highlight_borders,
                "opencv": self.artwork.opencv_highlight_borders,
                "handmade_path": "edges_handmade_sobel.jpg",
                "opencv_path": "edges_opencv_canny.jpg",
            },
            "gamma correction": {
                "handmade": lambda: self.artwork.handmade_gamma_correction(gamma_param),
                "opencv": lambda: self.artwork.opencv_gamma_correction(gamma_param),
                "handmade_path": "gamma_correction_handmade.jpg",
                "opencv_path": "gamma_correction_opencv.jpg",
            },
            "histogram equalization": {
                "handmade": self.artwork.handmade_histogram_equalization,
                "opencv": self.artwork.opencv_histogram_equalization,
                "handmade_path": "histogram_equalization_handmade.jpg",
                "opencv_path": "histogram_equalization_opencv.jpg",
            },
        }

        for name in tasks:
            self._apply_filter(name=name, options=tasks[name])

        log.info("Обработка завершена. Файлы сохранены в %s", self.save_path)
