import csv
import os
import random
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import config
from core.integrations.integration import download_files, make_request
from dataclass import BaseObject, MetObject
from logger import log


class AbstractFileProcessor(ABC):
    @abstractmethod
    def read_file(self, file: Path) -> list[MetObject]:
        pass
