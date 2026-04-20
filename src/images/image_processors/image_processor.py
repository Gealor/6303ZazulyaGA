from pathlib import Path
from string import Template

import config
from images.artwork import Artwork
from logger import log


class ImageProcessor:
    def __init__(self, artwork: Artwork, save_path: Path, id_image: int):
        self.artwork = artwork
        self.save_path = save_path
        self.filename_template = save_path.name
        self.id_image = id_image

    def _apply_filter(self, name: str, options: dict):
        """Унифицированное применение фильтров и сохранение"""
        log.info("[%d] Сравнение %s...", self.id_image, name)
        handmade = options["handmade"]()
        opencv2 = options["opencv"]()
        self.artwork.save_image(
            self.save_path / options["handmade_path"],
            handmade,
        )
        self.artwork.save_image(
            self.save_path / options["opencv_path"],
            opencv2,
        )

    def process_artwork(
        self,
        gamma_param: float = config.GAMMA_CORRECTION_PARAM,
        kernel_size: int = config.KERNEL_GAUSSIAN_SIZE,
    ):
        string_template = (
            self.filename_template + "_" + "${operation}" + "_" + "${type}" + ".jpg"
        )
        template = Template(string_template)
        tasks = {
            "grayscale": {
                "handmade": self.artwork.handmade_grayscale,
                "opencv": self.artwork.opencv_grayscale,
                "handmade_path": template.substitute(operation="gray", type="handmade"),
                "opencv_path": template.substitute(operation="gray", type="opencv"),
            },
            "gaussian blur": {
                "handmade": lambda: self.artwork.handmade_gaussian_blur(
                    kernel_size=kernel_size,
                ),
                "opencv": lambda: self.artwork.opencv_gaussian_blur(
                    kernel_size=kernel_size,
                ),
                "handmade_path": template.substitute(operation="blur", type="handmade"),
                "opencv_path": template.substitute(operation="blur", type="opencv"),
            },
            "edges": {
                "handmade": self.artwork.handmade_highlight_borders,
                "opencv": self.artwork.opencv_highlight_borders,
                "handmade_path": template.substitute(
                    operation="edges_sobel", type="handmade"
                ),
                "opencv_path": template.substitute(
                    operation="edges_canny", type="opencv"
                ),
            },
            "gamma correction": {
                "handmade": lambda: self.artwork.handmade_gamma_correction(gamma_param),
                "opencv": lambda: self.artwork.opencv_gamma_correction(gamma_param),
                "handmade_path": template.substitute(
                    operation="gamma_correction", type="handmade"
                ),
                "opencv_path": template.substitute(
                    operation="gamma_correction", type="opencv"
                ),
            },
            "histogram equalization": {
                "handmade": self.artwork.handmade_histogram_equalization,
                "opencv": self.artwork.opencv_histogram_equalization,
                "handmade_path": template.substitute(
                    operation="histogram_equalization", type="handmade"
                ),
                "opencv_path": template.substitute(
                    operation="histogram_equalization", type="opencv"
                ),
            },
        }

        for name in tasks:
            self._apply_filter(name=name, options=tasks[name])

        log.info(
            "[%d] Обработка завершена. Файлы сохранены в %s",
            self.id_image,
            self.save_path,
        )
