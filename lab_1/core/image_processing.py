from pathlib import Path
import cv2
import numpy as np

from decorators import time_meter_decorator
from logger import log
import config


print(np.sum(config.KERNEL_GAUSSIAN))


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
    h, w, _ = img.shape

    gray = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]

    return gray.astype(np.uint8)


@time_meter_decorator
# TODO: сделать возможность применения свертки к оригинальному изображению с тремя каналами RGB
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


@time_meter_decorator
def handmade_gaussian_blur(img: np.ndarray) -> np.ndarray:
    """
    Сглаживание Гаусса
    """
    # Ядро Гаусса 5x5 (аппроксимация)
    kernel = config.KERNEL_GAUSSIAN / np.sum(config.KERNEL_GAUSSIAN)  # 273.0
    return handmade_convolution(img, kernel)


@time_meter_decorator
def handmade_sobel(img: np.ndarray) -> np.ndarray:
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
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    grad_x = handmade_convolution(img, kx).astype(np.float32)
    grad_y = handmade_convolution(img, ky).astype(np.float32)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)


@time_meter_decorator
def opencv_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


@time_meter_decorator
def opencv_filter2D(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(img, -1, kernel)


@time_meter_decorator
def opencv_canny(img: np.ndarray) -> np.ndarray:
    return cv2.Canny(img, 100, 200)


@time_meter_decorator
def opencv_harris(img: np.ndarray) -> np.ndarray:
    # Детектор углов Харриса
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # Результат Харриса — это карта откликов, наложим её на картинку для визуализации
    img_copy = img.copy()
    img_copy[dst > 0.01 * dst.max()] = [0, 0, 255]
    return img_copy


def process_image(path: Path, name_original: str):
    original_path = path / name_original
    img = load_image(original_path)

    log.info("Сравнение grayscale...")
    gray_handmade = handmade_grayscale(img)
    gray_cv2 = opencv_grayscale(img)
    cv2.imwrite(path / "gray_handmade.jpg", gray_handmade)
    cv2.imwrite(path / "gray_opencv.jpg", gray_cv2)

    log.info("Сравнение размытия Гаусса...")
    blur_handmade = handmade_gaussian_blur(img)
    blur_cv2 = opencv_filter2D(
        img,
        config.KERNEL_GAUSSIAN / np.sum(config.KERNEL_GAUSSIAN),
    )
    cv2.imwrite(path / "blur_handmade.jpg", blur_handmade)
    cv2.imwrite(path / "blur_opencv.jpg", blur_cv2)

    log.info("Сравнение выделения границ...")
    edges_handmade = handmade_sobel(img)
    edges_canny = opencv_canny(img)
    cv2.imwrite(path / "edges_handmade_sobel.jpg", edges_handmade)
    cv2.imwrite(path / "edges_opencv_canny.jpg", edges_canny)

    log.info("Обработка завершена. Файлы сохранены в %s", path)
