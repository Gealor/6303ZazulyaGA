from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

import config
from core.exceptions import ShapeArtworkColorfulException
from dataclass import ImageObject
from decorators import time_meter_decorator
from logger import log


class Artwork(ABC):
    __slots__ = ('_path_file', '_img', '_name')

    def __init__(self, path: Path):
        self._path_file = path
        self._img = self._load_image(path)
        self._name = path.name

    @property
    def image(self):
        return self._img

    @property
    def path(self):
        return self._path_file

    @property
    def name(self):
        return self._name

    def _load_image(self, path: Path) -> np.ndarray:
        _img = cv2.imread(path)
        if _img is None:
            log.error("Не удалось загрузить изображение")
            raise ValueError

        log.info("Форма изображения: %s", _img.shape)
        return _img


    def _calculate_cdf(self, _img_channel: np.ndarray) -> np.ndarray:
        """Вспомогательная функция для расчета нормализованной CDF"""
        # гистограмма (сколько раз встречается каждое значение от 0 до 255)
        hist, _ = np.histogram(_img_channel.flatten(), bins=256, range=(0, 256))

        # накопленная сумма (CDF)
        cdf = hist.cumsum()

        # маскирование нулей (чтобы минимум был не 0)
        cdf_m = np.ma.masked_equal(cdf, 0)

        # нормализация CDF по формуле выравнивания гистограммы
        # (cdf - cdf_min) * 255 / (total_pixels - cdf_min)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())

        # возвращение нулей обратно в результирующую матрицу
        cdf = np.ma.filled(cdf_m, 0).astype(dtype=np.uint8)
        return cdf


    def save_image(self, img: np.ndarray, path: Path):
        log.info("Сохранение изображения в %s...", path)
        cv2.imwrite(path, img)


    @abstractmethod
    def handmade_grayscale(self) -> np.ndarray:
        pass


    @abstractmethod
    def handmade_convolution(self, kernel: np.ndarray) -> np.ndarray:
        pass


    @abstractmethod
    def handmade_histogram_equalization(self) -> np.ndarray:
        pass


    @abstractmethod
    def opencv_grayscale(self) -> np.ndarray:
        pass


    @abstractmethod
    def opencv_histogram_equalization(self) -> np.ndarray:
        pass


    @time_meter_decorator
    def handmade_gaussian_blur(self) -> np.ndarray:
        """
        Сглаживание Гаусса
        """
        # Ядро Гаусса 5x5 (аппроксимация)
        kernel = config.KERNEL_GAUSSIAN / np.sum(config.KERNEL_GAUSSIAN)  # 273.0
        return self.handmade_convolution(kernel)


    @time_meter_decorator
    def handmade_highlight_borders(self) -> np.ndarray:
        """
        Выделение границ с помощью оператора Собеля
        """
        # Оператор Собеля для границ
        kx = np.array(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ],
            dtype=np.float32,
        )
        ky = np.array(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ], dtype=np.float32
        )

        grad_x = self.handmade_convolution(kx).astype(np.float32)
        grad_y = self.handmade_convolution(ky).astype(np.float32)

        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return np.clip(magnitude, 0, 255).astype(np.uint8)


    @time_meter_decorator
    def handmade_gamma_correction(self, gamma: float) -> np.ndarray:
        """
        Гамма-коррекция.
        Если гамма > 1, то изображение становится темнее
        Если гамма < 1, то изображение становится светлее
        """
        inv_gamma = 1.0 / gamma
        _img_float = self._img.astype(dtype=np.float32) / 255.0
        corrected = np.power(_img_float, inv_gamma)

        return (corrected * 255).astype(dtype=np.uint8)


    @time_meter_decorator
    def opencv_filter2D(
        self,
        kernel: np.ndarray = config.KERNEL_GAUSSIAN / np.sum(config.KERNEL_GAUSSIAN)
    ) -> np.ndarray:
        # -1 значит, что глубина будет такой же, как и исходное изображение
        return cv2.filter2D(self._img, -1, kernel)


    @time_meter_decorator
    def opencv_gaussian_blur(self) -> np.ndarray:
        # 0 значит, что степень размытия определяется ядром
        return cv2.GaussianBlur(self._img, (5,5), 0)


    @time_meter_decorator
    def opencv_highlight_borders(self) -> np.ndarray:
        return cv2.Canny(self._img, 100, 200)


    @time_meter_decorator
    def opencv_gamma_correction(
        self,
        gamma: float = config.GAMMA_CORRECTION_PARAM,
    ) -> np.ndarray:
        """
        Гамма-коррекция через Look-Up Table (LUT) OpenCV.
        """
        inv_gamma = 1.0 / gamma
        # Массив (таблица) соответствия: индекс - старое значение пикселя -> новое значение пикселя
        table = np.array(
            [
                ((i / 255.0) ** inv_gamma) * 255
                for i in np.arange(0, 256)
            ]
        ).astype(dtype=np.uint8)

        # Применяю таблицу ко всему изображению
        return cv2.LUT(self._img, table)


    def __repr__(self):
        return f"{self.__class__.__name__}(path={self._path_file}, name={self._name})"

class ArtworkColorful(Artwork):
    # дублировать поля из родительского класса в slots не нужно
    __slots__ = ()

    def __init__(self, path: Path):
        super().__init__(path)
        if len(self._img.shape)!=3 or self._img.shape[2]!=3:
            log.error("Несоответствие количества каналов для цветного изображения")
            raise ShapeArtworkColorfulException

    @time_meter_decorator
    def handmade_grayscale(self) -> np.ndarray:
        """
        Перевод цветного изображения к grayscale
        """
        gray = (
            0.114 * self._img[:, :, 0] +
            0.587 * self._img[:, :, 1] +
            0.299 * self._img[:, :, 2]
        )

        return gray.astype(np.uint8)

    @time_meter_decorator
    def handmade_convolution(self, kernel: np.ndarray) -> np.ndarray:
        """
        Применений свертки к цветному изображению (размытие, резкость и т.д.)
        """
        h, w, _ = self._img.shape

        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        padding_config = ((pad_h, pad_h), (pad_w, pad_w), (0, 0))

        padded__img = np.pad(self._img, pad_width=padding_config, mode="constant")
        result = np.zeros_like(self._img, dtype=np.float32)

        for i in range(kh):
            for j in range(kw):
                region = padded__img[i : i + h, j : j + w, :]
                result += region * kernel[i, j]

        return np.clip(result, 0, 255).astype(dtype=np.uint8)

    @time_meter_decorator
    def handmade_histogram_equalization(self) -> np.ndarray:
        lab = cv2.cvtColor(self._img, cv2.COLOR_RGB2LAB)
        # Разделение каналов
        lightness, a, b = cv2.split(lab)
        # Выравнивание только L (Lightness) канала
        lut = self._calculate_cdf(lightness)

        l_eq = cv2.LUT(lightness, lut)
        merged = cv2.merge((l_eq, a, b))

        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    @time_meter_decorator
    def opencv_grayscale(self) -> np.ndarray:
        return cv2.cvtColor(self._img, cv2.COLOR_RGB2GRAY)

    @time_meter_decorator
    def opencv_histogram_equalization(self) -> np.ndarray:
        """
        Выравнивание гистограммы через opencv
        """
        lab = cv2.cvtColor(self._img, cv2.COLOR_RGB2LAB)
        channels = cv2.split(lab)
        cv2.equalizeHist(channels[0], channels[0])
        merged = cv2.merge(channels)
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    def __str__(self):
        return (
            f"{self.__class__.__name__}"
            f"(h={self._img.shape[0]}, w={self._img.shape[1]}, c={self._img.shape[2]})"
        )


class ArtworkGrayscale(Artwork):
    __slots__ = ()

    def __init__(self, path: Path):
        super().__init__(path)
        if len(self._img.shape) == 3:
            self._img = cv2.cvtColor(self._img, cv2.COLOR_RGB2GRAY).astype(dtype=np.uint8)
        self.save_image(self._img, path=path)

    @time_meter_decorator
    def handmade_grayscale(self) -> np.ndarray:
        return self._img

    @time_meter_decorator
    def handmade_convolution(self, kernel: np.ndarray) -> np.ndarray:
        """
        Применений свертки к цветному изображению (размытие, резкость и т.д.)
        """
        h, w = self._img.shape

        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        padding_config = ((pad_h, pad_h), (pad_w, pad_w))

        padded__img = np.pad(self._img, pad_width=padding_config, mode="constant")
        result = np.zeros_like(self._img, dtype=np.float32)

        for i in range(kh):
            for j in range(kw):
                region = padded__img[i : i + h, j : j + w]
                result += region * kernel[i, j]

        return np.clip(result, 0, 255).astype(dtype=np.uint8)

    @time_meter_decorator
    def handmade_histogram_equalization(self) -> np.ndarray:
        cdf = self._calculate_cdf(self._img)
        return cdf[self._img]

    @time_meter_decorator
    def opencv_grayscale(self) -> np.ndarray:
        return self._img

    @time_meter_decorator
    def opencv_histogram_equalization(self) -> np.ndarray:
        return cv2.equalizeHist(self._img)

    def __str__(self):
        return (
            f"{self.__class__.__name__}"
            f"(h={self._img.shape[0]}, w={self._img.shape[1]}, c=1)"
        )
