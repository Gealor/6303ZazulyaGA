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
KERNEL_GAUSSIAN = np.array(
    [
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1],
    ],
    dtype=np.float32,
)
KERNEL_GAUSSIAN_SIZE = 5
GAMMA_CORRECTION_PARAM = 0.5
