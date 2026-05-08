import asyncio
import random
from abc import ABC, abstractmethod
from pathlib import Path
import shutil
from typing import List, Tuple

import aiofiles.os
import aiohttp
import aioshutil

import memetl.config as config
from memetl.dataclass import BaseObject, MetObject
from memetl.decorators import async_time_meter_decorator
from memetl.images.integrations.async_integration import (
    make_request_and_save_info,
    download_files as async_download_files,
    semaphore_wrapper,
)
from memetl.images.integrations.integration import (
    make_request,
    download_files as sync_download_files,
)
from memetl.logger import log


class AbstractFileProcessor(ABC):
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

    @abstractmethod
    def read_file(self, file: Path) -> list[BaseObject]:
        pass

    def start_pipeline(
        self,
        read_file: Path = config.MET_OBJECTS_PATH,
        count: int = 1,
        classification: str = config.PAINTING_CLASSIFICATION,
        file_name: str = config.ORIGINAL_IMAGE,
    ) -> List[Tuple[Path, Path]]:
        objects = self.read_file(read_file)

        # Фильтрация объектов, по классификации, по умолчанию картинка
        random_objects = self.select_objects_sample(objects, count, classification)  # type: ignore

        return self.process_by_object_list(random_objects)

    def select_objects_sample(
        self,
        objects: list[MetObject],
        count: int,
        classification: str = config.PAINTING_CLASSIFICATION,
    ) -> list[MetObject]:
        """
        Фильтрация и составление выборки объектов, по классификации, по умолчанию картинка
        """
        log.info("Фильтрация данных...")
        filtered_objects = [
            elem for elem in objects if elem.classification == classification
        ]
        # Выбираю случайный объект
        log.info("Выбор %d случайных элементов...", count)
        random_objects = random.sample(filtered_objects, k=count)
        log.debug(
            "IDs выбранных объектов: %s",
            [random_object.object_id for random_object in random_objects],
        )

        return random_objects

    def _clear_folder(self):
        # Получаем абсолютные пути, чтобы раскрыть любые симлинки или "..", "."
        target_path = self.full_path.resolve()
        base_path = self.base_dir.resolve()

        # Убеждаемся, что мы удаляем папку СТРОГО внутри base_dir
        # Это защитит от path traversal атак (если save_folder = "../../windows")
        if not target_path.is_relative_to(base_path):
            log.critical("Попытка удалить директорию ВНЕ рабочей зоны: %s", target_path)
            raise PermissionError("Попытка удалить директорию вне разрешенной зоны!")

        # Запрещаем удалять саму базовую директорию
        # Это сработает, если save_folder = "" или "."
        if target_path == base_path:
            log.critical("Попытка удалить базовую рабочую директорию: %s", target_path)
            raise PermissionError("Нельзя удалять базовую директорию!")

        if target_path.exists():
            log.debug("Удаление папки %s...", target_path.as_posix())
            shutil.rmtree(target_path)

    def _create_dir(self, path: Path | None = None):
        """
        Создание директории path, если не указано, то создает базовую директорию
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

        sync_download_files(path=file_path, url=extended_object.primary_image)
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


class AbstractAsyncFileProcessor(ABC):
    def __init__(
        self,
        client_session: aiohttp.ClientSession,
        save_folder: str = config.PAINTINGS_DIR_NAME,
        base_dir: Path = config.WORK_DIR,
    ):
        self.save_folder = save_folder
        self.base_dir = base_dir
        self.client_session = client_session

    @property
    def full_path(self):
        return self.base_dir / self.save_folder

    @abstractmethod
    async def read_file(self, file: Path) -> list[BaseObject]:
        pass

    def select_objects_sample(
        self,
        objects: list[MetObject],
        count: int,
        classification: str = config.PAINTING_CLASSIFICATION,
    ) -> list[MetObject]:
        """
        Фильтрация и составление выборки объектов, по классификации, по умолчанию картинка
        """
        log.info("Фильтрация данных...")
        filtered_objects = [
            elem for elem in objects if elem.classification == classification
        ]
        # Выбираю случайный объект
        log.info("Выбор %d случайных элементов...", count)
        random_objects = random.sample(filtered_objects, k=count)
        log.debug(
            "IDs выбранных объектов: %s",
            [random_object.object_id for random_object in random_objects],
        )

        return random_objects

    async def _clear_folder(self):
        # Получаем абсолютные пути, чтобы раскрыть любые симлинки или "..", "."
        target_path = self.full_path.resolve()
        base_path = self.base_dir.resolve()

        # Убеждаемся, что мы удаляем папку СТРОГО внутри base_dir
        # Это защитит от path traversal атак (если save_folder = "../../windows")
        if not target_path.is_relative_to(base_path):
            log.critical("Попытка удалить директорию ВНЕ рабочей зоны: %s", target_path)
            raise PermissionError("Попытка удалить директорию вне разрешенной зоны!")

        # Запрещаем удалять саму базовую директорию
        # Это сработает, если save_folder = "" или "."
        if target_path == base_path:
            log.critical("Попытка удалить базовую рабочую директорию: %s", target_path)
            raise PermissionError("Нельзя удалять базовую директорию!")

        if target_path.exists():
            log.debug("Удаление папки %s...", target_path.as_posix())
            await aioshutil.rmtree(target_path)

    async def _create_dir(self, path: Path | None = None):
        """
        Создание директорию path, если не указано, то создает базовую директорию
        """
        if path is None:
            path = self.full_path
        if not path.exists():
            log.debug("Создание директории %s...", path.name)
            await aiofiles.os.mkdir(path=path)
        else:
            log.debug("Директория уже создана. Пропускаем...")

    async def _get_and_download(
        self, object_id: str, file_path: Path, dir_path: Path
    ) -> bool:
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

        await async_download_files(
            object_id=object_id,
            path=file_path,
            url=extended_object.primary_image,
            client_session=self.client_session,
        )
        return True

    async def _handle_one_element(
        self, index: int, obj: MetObject
    ) -> Tuple[Path, Path] | None:
        log.info("Обработка объекта #%d с ID = %s", index, obj.object_id)
        file_name, dir_name = (
            f"{index}_{obj.object_id}_{config.ORIGINAL_IMAGE}",
            f"{index}_{obj.object_id}",
        )
        dir_path = self.full_path / dir_name
        file_path = dir_path / file_name
        await self._create_dir(path=dir_path)
        success_download = await self._get_and_download(
            object_id=obj.object_id,
            file_path=file_path,
            dir_path=dir_path,
        )
        if success_download:
            return file_path, dir_path

        log.info("Объект %s обработан.\n", file_name)

    async def process_by_object_list(
        self, objects: list[MetObject]
    ) -> List[Tuple[Path, Path]]:
        """Запускает процесс скачивания и получения информации об изображениях по списку"""
        await self._clear_folder()
        await self._create_dir()
        semaphore = asyncio.Semaphore(value=config.SEMAPHORE_COUNT)
        list_coros = [
            semaphore_wrapper(self._handle_one_element(index, obj), semaphore)
            for index, obj in enumerate(objects, start=1)
        ]
        results = await asyncio.gather(*list_coros)
        return [result for result in results if result is not None]

    @async_time_meter_decorator
    async def start_pipeline(
        self,
        read_file: Path = config.MET_OBJECTS_PATH,
        count: int = 1,
        classification: str = config.PAINTING_CLASSIFICATION,
    ) -> List[Tuple[Path, Path]]:
        objects = await self.read_file(read_file)

        # Фильтрация объектов, по классификации, по умолчанию картинка
        random_objects = self.select_objects_sample(objects, count, classification)  # type: ignore

        return await self.process_by_object_list(random_objects)
