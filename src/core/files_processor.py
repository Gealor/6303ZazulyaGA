import csv
import os
import random
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import config
from core.integration import download_files, make_request
from dataclass import BaseObject, MetObject
from logger import log

random.seed(52)


class AbstractFileProcessor(ABC):
    __slots__ = ("save_folder", "base_dir")

    def __init__(
        self,
        save_folder: str = config.PAINTINGS_DIR_NAME,
        base_dir: Path = config.BASE_DIR,
    ):
        self.save_folder = save_folder
        self.base_dir = base_dir

    @property
    def full_path(self):
        return self.base_dir / self.save_folder

    @abstractmethod
    def read_file(self, file: Path) -> list[MetObject]:
        pass

    def _clear_folder(self):
        if self.full_path.exists():
            log.info("Удаление папки %s...", self.full_path.as_posix())
            shutil.rmtree(self.full_path)

    def _create_dir(self):
        """
        Создание нужной директории с именем name
        """
        if not os.path.exists(self.full_path):
            log.info("Создание директории %s...", self.save_folder)
            os.makedirs(self.full_path)
        else:
            log.info("Директория уже создана. Пропускаем...")

    def _get_and_download(
        self, object_id: str, file_name: str = config.ORIGINAL_IMAGE
    ) -> Path:
        extended_object = make_request(object_id)
        file_path = self.full_path / file_name
        download_files(path=self.full_path / file_name, url=extended_object.primary_image)
        return file_path

    def start_pipeline(
        self,
        read_file: Path,
        classification: str = config.PAINTING_CLASSIFICATION,
        file_name: str = config.ORIGINAL_IMAGE,
    ) -> Tuple[Path, Path]:
        self._clear_folder()
        self._create_dir()
        objects = self.read_file(read_file)

        # Фильтрация объектов, по классификации, по умолчанию картинка
        log.info("Фильтрация данных...")
        filtered_objects = [
            elem for elem in objects if elem.classification == classification
        ]
        # Выбираю случайный объект
        log.info("Выбор случайного элемента...")
        random_object = random.choice(filtered_objects)
        log.info("Выбран объект с ID = %s", random_object.object_id)

        log.info("Запрос к стороннему API...")
        saved_file_path = self._get_and_download(
            object_id=random_object.object_id, file_name=file_name
        )

        return saved_file_path, saved_file_path.parent


class CSVFileProcessor(AbstractFileProcessor):
    __slots__ = ()

    def start_pipeline(
        self,
        read_file: Path = config.MET_OBJECTS_FILE,
        classification: str = config.PAINTING_CLASSIFICATION,
        file_name: str = config.ORIGINAL_IMAGE,
    ) -> Tuple[Path, Path]:
        return super().start_pipeline(read_file, classification, file_name)

    def read_file(self, file: Path = config.MET_OBJECTS_FILE) -> list[BaseObject]:
        """
        Чтение .csv файла и получение всех объектов с их идентификаторами и классификациями(классами)
        """
        result = []
        log.info("Чтение .csv файла...")
        with open(
            file,
            mode="r",
            encoding="utf-8-sig",
        ) as f:  # sig, чтобы убрать \ufeff символ
            try:
                csv_reader = csv.DictReader(f)
            except Exception as e:
                log.error("Ошибка при чтении csv файла: %s", e)
                raise e

            for row in csv_reader:
                obj = MetObject(
                    object_id=row["Object ID"],
                    classification=row["Classification"],
                )
                result.append(obj)

        log.info("Файл прочитан успешно.")
        return result
