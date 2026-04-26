import csv
import shutil
from pathlib import Path
from typing import List, Tuple

import memetl.config as config
from memetl.dataclass import MetObject
from memetl.images.exceptions import IncorrectFormatCSVException
from memetl.images.file_processors.abstract_file_processor import AbstractFileProcessor
from memetl.images.integrations.integration import download_files, make_request
from memetl.logger import log


class CSVFileProcessor(AbstractFileProcessor):
    def __init__(
        self,
        save_folder: str = config.PAINTINGS_DIR_NAME,
        base_dir: Path = config.WORK_DIR,
    ):
        self.save_folder = save_folder
        self.base_dir = base_dir

    @property
    def full_path(self):
        return self.base_dir / self.save_folder

    def _clear_folder(self):
        if self.full_path.exists():
            log.debug("Удаление папки %s...", self.full_path.as_posix())
            shutil.rmtree(self.full_path)

    def _create_dir(self, path: Path | None = None):
        """
        Создание директорию path, если не указано, то создает базовую директорию
        """
        if path is None:
            path = self.full_path
        if not path.exists():
            log.debug("Создание директории %s...", path.name)
            path.mkdir(parents=True, exist_ok=True)
        else:
            log.debug("Директория уже создана. Пропускаем...")

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

    def process_by_object_list(self, objects: list[MetObject]) -> List[Tuple[Path, Path]]:
        """Запускает процесс скачивания и получения информации об изображениях по списку"""
        self._clear_folder()
        self._create_dir()
        results = []
        for index, obj in enumerate(objects, start=1):
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

    def read_file(self, file: Path | str = config.MET_OBJECTS_PATH) -> list[MetObject]:
        """
        Чтение .csv файла и получение всех объектов с их идентификаторами и классификациями(классами)
        """
        result = []
        log.info("Чтение .csv файла...")
        with open(
            file, mode="r", encoding="utf-8-sig"
        ) as f:  # sig, чтобы убрать \ufeff символ
            try:
                csv_reader = csv.DictReader(f)
            except Exception as e:
                log.error("Ошибка при чтении csv файла: %s", e)
                raise

            for row in csv_reader:
                try:
                    if row["Is Public Domain"] == "True":
                        obj = MetObject(
                            object_id=row["Object ID"],
                            classification=row["Classification"],
                        )
                        result.append(obj)
                except KeyError as e:
                    log.warning("Ошибка при доступе к аттрибуту: %s", e)
                    raise IncorrectFormatCSVException from KeyError
        log.info("Файл прочитан успешно.")
        return result

    def start_pipeline(
        self,
        read_file: Path = config.MET_OBJECTS_PATH,
        count: int = 1,
        classification: str = config.PAINTING_CLASSIFICATION,
        file_name: str = config.ORIGINAL_IMAGE,
    ) -> List[Tuple[Path, Path]]:
        objects = self.read_file(read_file)

        # Фильтрация объектов, по классификации, по умолчанию картинка
        random_objects = self.select_objects_sample(objects, count, classification)

        return self.process_by_object_list(random_objects)
