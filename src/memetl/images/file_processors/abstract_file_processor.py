import csv
import os
import random
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import memetl.config as config
from memetl.images.integrations.integration import download_files, make_request
from memetl.dataclass import BaseObject, MetObject
from memetl.logger import log


class AbstractFileProcessor(ABC):
    @abstractmethod
    def read_file(self, file: Path) -> list[MetObject]:
        pass
