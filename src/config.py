from pathlib import Path

import numpy as np

# Files and dirs
PAINTINGS_DIR_NAME = "paintings"
METADATA_FILE = "metadata.json"
BASE_DIR = Path(__file__).parent

MET_OBJECTS_FILE = BASE_DIR / "MetObjects.csv"
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
