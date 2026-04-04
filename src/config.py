from pathlib import Path

import numpy as np

# Files and dirs
PAINTINGS_DIR_NAME = "paintings"
METADATA_FILE = "metadata.json"
BAR_FIG_NAME = "столбчатая_диаграмма.png"
SLIDE_WINDOW_NAME = "график_скользящее_окно.png"

BASE_DIR = Path(__file__).parent

MET_OBJECTS_PATH = BASE_DIR / "MetObjects.csv"
BAR_FIG_PATH = BASE_DIR / BAR_FIG_NAME
SLIDE_WINDOW_PATH = BASE_DIR / SLIDE_WINDOW_NAME
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
