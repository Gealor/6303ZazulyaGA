from pathlib import Path

import numpy as np

# Files and dirs
PAINTINGS_DIR_NAME = "paintings"
METADATA_FILE = "metadata.json"
BAR_FIG_NAME = "столбчатая_диаграмма.png"
SLIDE_WINDOW_NAME = "график_скользящее_окно.png"

# Базовые пути проекта
BASE_DIR = Path(__file__).parent  # src/memetl
PACKAGE_DIR = BASE_DIR.parent.parent  # папка проекта (где src/)


WORK_DIR = Path.cwd()
MET_OBJECTS_PATH = WORK_DIR / "data" / "MetObjects.csv"
# Пути для сохранения результатов (в текущую рабочую директорию)
PAINTINGS_DIR = WORK_DIR / "paintings"
ANALYSIS_RESULTS_PATH = WORK_DIR / "analysis_results"
LOG_DIR_PATH = WORK_DIR / "logs"

PAINTING_CLASSIFICATION = "Paintings"
ORIGINAL_IMAGE = "original.jpg"

# Integration config
BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1/objects/"


# Image processing
KERNEL_GAUSSIAN_SIZE = 11
GAMMA_CORRECTION_PARAM = 0.5
COEF_ADDING = 0.3


# Logger configuration
FORMAT = "%(asctime)s | %(module)s | %(levelname)s | %(message)s"
DATEFMT = "%Y-%m-%d %H-%M-%S"


# Defaults
DEFAULT_IMAGE_NAME = "image"


# Async params
SEMAPHORE_COUNT = 10
