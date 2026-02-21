from pathlib import Path

import cv2
import numpy as np

import config
from decorators import time_meter_decorator
from logger import log


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        log.error("Не удалось загрузить изображение")
        raise ValueError

    log.info("Форма изображения: %s", img.shape)
    return img


@time_meter_decorator
def handmade_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Перевод изображения к grayscale
    """
    coef = np.array([0.114, 0.587, 0.299])
    gray = np.sum(coef * img, axis = 2)
    # gray = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]

    return gray.astype(np.uint8)


@time_meter_decorator
def handmade_convolution(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Применений свертки к изображению (размытие, резкость и т.д.)
    """
    if len(img.shape) == 3:
        h, w, _ = img.shape
        is_rgb = True
    else:
        h, w = img.shape
        is_rgb = False

    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padding_config = (
        ((pad_h, pad_h), (pad_w, pad_w), (0, 0))
        if is_rgb
        else ((pad_h, pad_h), (pad_w, pad_w))
    )

    # добавляю рамку вокруг изображения
    padded_img = np.pad(img, pad_width=padding_config, mode="constant")
    result = np.zeros_like(img, dtype=np.float32)

    # Итерация по ядру, т.к. это быстрее, чем перебирать ВСЕ пиксели изображения
    for i in range(kh):
        for j in range(kw):
            region = (
                padded_img[i : i + h, j : j + w, :]
                if is_rgb
                else padded_img[i : i + h, j : j + w]
            )
            result += region * kernel[i, j]

    return np.clip(result, 0, 255).astype(dtype=np.uint8)


def create_gaussian_kernel(
    size: int, sigma: float | None = None, normalize: bool = True
) -> np.ndarray:
    """Создать ядро Гаусса размерности size на size"""
    if size % 2 == 0:
        raise ValueError("Размер ядра должен быть нечетным")

    if sigma is None:
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8

    center = size // 2
    x = np.linspace(-center, center, size)
    y = np.linspace(-center, center, size)
    x, y = np.meshgrid(x, y)

    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    if normalize:
        kernel = kernel / np.sum(kernel)

    return kernel


@time_meter_decorator
def handmade_gaussian_blur(
    img: np.ndarray, kernel_size: int = config.KERNEL_GAUSSIAN_SIZE
) -> np.ndarray:
    """
    Сглаживание Гаусса
    """
    # Ядро Гаусса nxn (аппроксимация)
    kernel = create_gaussian_kernel(kernel_size)

    return handmade_convolution(img, kernel)


@time_meter_decorator
def handmade_sobel(img: np.ndarray) -> np.ndarray:
    """
    Выделение границ с помощью оператора Собеля
    """
    if len(img.shape) == 3:
        img = handmade_grayscale(img)
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
            [1, 2, 1],
        ],
        dtype=np.float32,
    )

    grad_x = handmade_convolution(img, kx).astype(np.float32)
    grad_y = handmade_convolution(img, ky).astype(np.float32)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude[magnitude > 128] = 255
    magnitude[magnitude <= 128] = 0
    return magnitude.astype(np.uint8)


@time_meter_decorator
def handmade_gamma_correction(
    img: np.ndarray,
    gamma: float = config.GAMMA_CORRECTION_PARAM,
) -> np.ndarray:
    """
    Гамма-коррекция.
    Если гамма > 1, то изображение становится темнее
    Если гамма < 1, то изображение становится светлее
    """
    inv_gamma = 1.0 / gamma
    img_float = img.astype(dtype=np.float32) / 255.0
    corrected = np.power(img_float, inv_gamma)

    return (corrected * 255).astype(dtype=np.uint8)


@time_meter_decorator
def handmade_histogram_equalization(img: np.ndarray) -> np.ndarray:
    """Ручное выравнивание гистограммы"""

    def _calculate_cdf(img_channel: np.ndarray) -> np.ndarray:
        """Вспомогательная функция для расчета нормализованной CDF"""
        # гистограмму (сколько раз встречается каждое значение от 0 до 255)
        hist, _ = np.histogram(img_channel.flatten(), bins=256, range=(0, 256))

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

    if len(img.shape) == 2:
        cdf = _calculate_cdf(img)
        return cdf[img]
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        # Разделение каналов
        lightness, a, b = cv2.split(lab)
        # Выравнивание только L (Lightness) канала
        lut = _calculate_cdf(lightness)

        l_eq = cv2.LUT(lightness, lut)
        merged = cv2.merge((l_eq, a, b))

        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


@time_meter_decorator
def opencv_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


@time_meter_decorator
def opencv_filter2D(
    img: np.ndarray,
    kernel: np.ndarray = config.KERNEL_GAUSSIAN,
) -> np.ndarray:
    return cv2.filter2D(img, -1, kernel)


@time_meter_decorator
def opencv_gaussian_blur(
    img: np.ndarray,
    kernel_size: int = config.KERNEL_GAUSSIAN_SIZE,
) -> np.ndarray:
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


@time_meter_decorator
def opencv_canny(img: np.ndarray) -> np.ndarray:
    return cv2.Canny(img, 100, 200)


@time_meter_decorator
def opencv_gamma_correction(
    img: np.ndarray,
    gamma: float = config.GAMMA_CORRECTION_PARAM,
) -> np.ndarray:
    """
    Гамма-коррекция через Look-Up Table (LUT) OpenCV.
    """
    inv_gamma = 1.0 / gamma
    # Массив (таблица) соответствия: индекс - старое значение пикселя -> новое значение пикселя
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)],
    ).astype(dtype=np.uint8)

    # Применяю таблицу ко всему изображению
    return cv2.LUT(img, table)


@time_meter_decorator
def opencv_histogram_equalization(img: np.ndarray) -> np.ndarray:
    """
    Выравнивание гистограммы через opencv
    """
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        channels = cv2.split(lab)
        cv2.equalizeHist(channels[0], channels[0])
        merged = cv2.merge(channels)
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


methods = {
    "grayscale": {
        "handmade": handmade_grayscale,
        "opencv": opencv_grayscale,
        "handmade_path": "gray_handmade.jpg",
        "opencv_path": "gray_opencv.jpg",
    },
    "gaussian blur": {
        "handmade": handmade_gaussian_blur,
        "opencv": opencv_gaussian_blur,
        "handmade_path": "blur_handmade.jpg",
        "opencv_path": "blur_opencv.jpg",
    },
    "edges": {
        "handmade": handmade_sobel,
        "opencv": opencv_canny,
        "handmade_path": "edges_handmade_sobel.jpg",
        "opencv_path": "edges_opencv_canny.jpg",
    },
    "gamma correction": {
        "handmade": handmade_gamma_correction,
        "opencv": opencv_gamma_correction,
        "handmade_path": "gamma_correction_handmade.jpg",
        "opencv_path": "gamma_correction_opencv.jpg",
    },
    "histogram equalization": {
        "handmade": handmade_histogram_equalization,
        "opencv": opencv_histogram_equalization,
        "handmade_path": "histogram_equalization_handmade.jpg",
        "opencv_path": "histogram_equalization_opencv.jpg",
    },
}


def _apply_filter(name: str, img: np.ndarray, path: Path):
    """Унифицированное применение фильтра и сохранение"""
    log.info("Сравнение %s...", name)
    handmade = methods[name]["handmade"](img)
    opencv2 = methods[name]["opencv"](img)
    cv2.imwrite(path / methods[name]["handmade_path"], handmade)
    cv2.imwrite(path / methods[name]["opencv_path"], opencv2)


def process_image(path: Path, name_original: str):
    original_path = path / name_original
    img = load_image(original_path)

    for name in methods:
        _apply_filter(name=name, img=img, path=path)

    log.info("Обработка завершена. Файлы сохранены в %s", path)
