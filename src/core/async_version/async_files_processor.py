import asyncio
import csv
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import aiofiles.os
import aiohttp
import aioshutil

import config
from core.async_version.async_integration import (
    download_files,
    make_request_and_save_info,
)
from dataclass import BaseObject, MetObject
from logger import log


class AbstractAsyncFileProcessor(ABC):
    def __init__(
        self,
        client_session: aiohttp.ClientSession,
        save_folder: str = config.PAINTINGS_DIR_NAME,
        base_dir: Path = config.BASE_DIR,
    ):
        self.save_folder = save_folder
        self.base_dir = base_dir
        self.client_session = client_session

    @property
    def full_path(self):
        return self.base_dir / self.save_folder

    @abstractmethod
    def read_file(self, file: Path) -> list[MetObject]:
        pass

    async def _clear_folder(self):
        if self.full_path.exists():
            log.info("Удаление папки %s...", self.full_path.as_posix())
            await aioshutil.rmtree(self.full_path)

    async def _create_dir(self, path: Path | None = None):
        """
        Создание директорию path, если не указано, то создает базовую директорию
        """
        if path is None:
            path = self.full_path
        if not path.exists():
            log.info("Создание директории %s...", path.name)
            await aiofiles.os.mkdir(path = path)
        else:
            log.info("Директория уже создана. Пропускаем...")

    async def _get_and_download(self, object_id: str, file_path: Path, dir_path: Path) -> bool:
        metadata_path = dir_path / config.METADATA_FILE
        extended_object = await make_request_and_save_info(
            object_id,
            metadata_path=metadata_path,
            client_session=self.client_session,
        )
        if not extended_object.primary_image:
            log.warning(
                "Объект с ID=%s не содержит ссылки на скачивание. Пропускаем этот файл",
                extended_object.object_id,
            )
            return False

        await download_files(
            object_id=object_id,
            path=file_path,
            url=extended_object.primary_image,
            client_session=self.client_session,
        )
        return True


    async def _handle_one_element(self, index: int, obj: MetObject) -> Tuple[Path, Path] | None:
        log.info("Обработка объекта #%d с ID = %s", index, obj.object_id)
        file_name, dir_name = (
            f"{index}_{obj.object_id}_{config.ORIGINAL_IMAGE}",
            f"{index}_{obj.object_id}",
        )
        dir_path = self.full_path / dir_name
        file_path = dir_path / file_name
        await self._create_dir(path = dir_path)
        success_download = await self._get_and_download(
            object_id=obj.object_id,
            file_path=file_path,
            dir_path=dir_path,
        )
        if success_download:
            return file_path, dir_path

        log.info("Объект %s обработан.\n", file_name)


    async def start_pipeline(
        self,
        read_file: Path,
        count: int = 1,
        classification: str = config.PAINTING_CLASSIFICATION,
        file_name: str = config.ORIGINAL_IMAGE,
    ) -> List[Tuple[Path, Path]]:
        await self._clear_folder()
        await self._create_dir()
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
            [random_object.object_id for random_object in random_objects]
        )

        list_coros = [
            self._handle_one_element(index, obj) 
            for index, obj in enumerate(random_objects, start=1)
        ]
        results = await asyncio.gather(*list_coros)
        filtered_results = [result for result in results if result is not None]

        return filtered_results


class CSVAsyncFileProcessor(AbstractAsyncFileProcessor):
    async def start_pipeline(
        self,
        read_file: Path = config.MET_OBJECTS_PATH,
        count: int = 1,
        classification: str = config.PAINTING_CLASSIFICATION,
        file_name: str = config.ORIGINAL_IMAGE,
    ) -> List[Tuple[Path, Path]]:
        return await super().start_pipeline(
            read_file=read_file,
            count=count,
            classification=classification,
            file_name=file_name,
        )

    def read_file(self, file: Path = config.MET_OBJECTS_PATH) -> list[BaseObject]:
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
                raise

            for row in csv_reader:
                obj = MetObject(
                    object_id=row["Object ID"],
                    classification=row["Classification"],
                )
                result.append(obj)

        log.info("Файл прочитан успешно.")
        return result