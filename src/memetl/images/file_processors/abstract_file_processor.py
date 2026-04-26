import csv
import os
import random
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import memetl.config as config
from memetl.dataclass import BaseObject, MetObject
from memetl.images.integrations.integration import download_files, make_request
from memetl.logger import log


class AbstractFileProcessor(ABC):
    @abstractmethod
    def read_file(self, file: Path) -> list[BaseObject]:
        pass

    @abstractmethod
    def start_pipeline(
        self, read_file: Path, count: int, classification: str, file_name: str
    ) -> List[Tuple[Path, Path]]:
        pass


    def select_objects_sample(
        self,
        objects: list[MetObject],
        count: int,
        classification: str = config.PAINTING_CLASSIFICATION,
    ) -> list[MetObject]:
        '''
        Фильтрация и составление выборки объектов, по классификации, по умолчанию картинка
        '''
        log.info("Фильтрация данных...")
        filtered_objects = [
            elem for elem in objects if elem.classification == classification
        ]
        # Выбираю случайный объект
        log.info("Выбор %d случайных элементов...", count)
        random_objects = random.sample(filtered_objects, k=count)
        log.debug(
            "IDs выбранных объектов: %s",
            [random_object.object_id for random_object in random_objects]
        )

        return random_objects


class AbstractAsyncFileProcessor(ABC):
    @abstractmethod
    async def read_file(self, file: Path) -> list[BaseObject]:
        pass

    @abstractmethod
    async def start_pipeline(
        self, read_file: Path, count: int, classification: str, file_name: str
    ) -> List[Tuple[Path, Path]]:
        pass

    def select_objects_sample(
        self,
        objects: list[MetObject],
        count: int,
        classification: str = config.PAINTING_CLASSIFICATION,
    ) -> list[MetObject]:
        '''
        Фильтрация и составление выборки объектов, по классификации, по умолчанию картинка
        '''
        log.info("Фильтрация данных...")
        filtered_objects = [
            elem for elem in objects if elem.classification == classification
        ]
        # Выбираю случайный объект
        log.info("Выбор %d случайных элементов...", count)
        random_objects = random.sample(filtered_objects, k=count)
        log.debug(
            "IDs выбранных объектов: %s",
            [random_object.object_id for random_object in random_objects]
        )

        return random_objects
