import csv
import os
import random
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import config
from images.file_processors.csv.base_csv_file_processor import BaseCSVFileProcessor
from images.integrations.integration import download_files, make_request
from dataclass import BaseObject, MetObject
from logger import log


class CSVFileProcessor(BaseCSVFileProcessor):
    def __init__(
        self,
        save_folder: str = config.PAINTINGS_DIR_NAME,
        base_dir: Path = config.ROOT_DIR,
    ):
        self.save_folder = save_folder
        self.base_dir = base_dir

    @property
    def full_path(self):
        return self.base_dir / self.save_folder

    def _clear_folder(self):
        if self.full_path.exists():
            log.info("Удаление папки %s...", self.full_path.as_posix())
            shutil.rmtree(self.full_path)

    def _create_dir(self, path: Path | None = None):
        """
        Создание директорию path, если не указано, то создает базовую директорию
        """
        if path is None:
            path = self.full_path
        if not path.exists():
            log.info("Создание директории %s...", path.name)
            path.mkdir(parents=True, exist_ok=True)
        else:
            log.info("Директория уже создана. Пропускаем...")

    def _get_and_download(
        self,
        object_id: str,
        file_path: Path,
        dir_path: Path,
    ) -> bool:
        metadata_path = dir_path / config.METADATA_FILE
        extended_object = make_request(object_id, metadata_path=metadata_path)
        if not extended_object.primary_image:
            log.warning(
                "Объект с ID=%s не содержит ссылки на скачивание. Пропускаем этот файл",
                extended_object.object_id,
            )
            return False

        download_files(path=file_path, url=extended_object.primary_image)
        return True

    def start_pipeline(
        self,
        read_file: Path = config.MET_OBJECTS_PATH,
        count: int = 1,
        classification: str = config.PAINTING_CLASSIFICATION,
        file_name: str = config.ORIGINAL_IMAGE,
    ) -> List[Tuple[Path, Path]]:
        self._clear_folder()
        self._create_dir()
        objects = self.read_file(read_file)

        # Фильтрация объектов, по классификации, по умолчанию картинка
        log.info("Фильтрация данных...")
        filtered_objects = [
            elem for elem in objects if elem.classification == classification
        ]
        # Выбираю случайный объект
        log.info("Выбор %d случайных элементов...", count)
        random_objects = random.sample(filtered_objects, k=count)
        log.info(
            "IDs выбранных объектов: %s",
            [random_object.object_id for random_object in random_objects],
        )
        results = []
        for index, obj in enumerate(random_objects, start=1):
            log.info("Обработка объекта #%d с ID = %s", index, obj.object_id)
            file_name, dir_name = (
                f"{index}_{obj.object_id}_{config.ORIGINAL_IMAGE}",
                f"{index}_{obj.object_id}",
            )
            dir_path = self.full_path / dir_name
            file_path = dir_path / file_name
            self._create_dir(path=dir_path)
            success_download = self._get_and_download(
                object_id=obj.object_id,
                file_path=file_path,
                dir_path=dir_path,
            )
            if success_download:
                results.append((file_path, dir_path))

            log.info("Объект %s обработан.\n", file_name)

        return results
