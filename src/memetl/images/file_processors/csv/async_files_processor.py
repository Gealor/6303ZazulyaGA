import asyncio
import csv
from pathlib import Path
from typing import List, Tuple

import aiofiles.os
import aiohttp
import aioshutil

import memetl.config as config
from memetl.dataclass import MetObject
from memetl.decorators import async_time_meter_decorator
from memetl.images.exceptions import IncorrectFormatCSVException
from memetl.images.file_processors.abstract_file_processor import (
    AbstractAsyncFileProcessor,
)
from memetl.images.integrations.async_integration import (
    download_files,
    make_request_and_save_info,
    semaphore_wrapper,
)
from memetl.logger import log


class CSVAsyncFileProcessor(AbstractAsyncFileProcessor):
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

    async def _clear_folder(self):
        if self.full_path.exists():
            log.debug("Удаление папки %s...", self.full_path.as_posix())
            await aioshutil.rmtree(self.full_path)

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

        await download_files(
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

    async def read_file(
        self, file: Path | str = config.MET_OBJECTS_PATH
    ) -> list[MetObject]:
        """
        Чтение .csv файла и получение всех объектов с их идентификаторами и классификациями(классами)
        """
        result = []
        log.info("Чтение .csv файла...")
        try:
            async with aiofiles.open(file, mode="r", encoding="utf-8-sig") as f:
                content = await f.read()
        except Exception as e:
            log.error("Ошибка при чтении csv файла: %s", e)
            raise

        try:
            csv_reader = csv.DictReader(content.splitlines())
        except Exception as e:
            log.error("Ошибка при парсинге csv: %s", e)
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
                raise IncorrectFormatCSVException from e

        log.info("Файл прочитан успешно.")
        return result

    @async_time_meter_decorator
    async def start_pipeline(
        self,
        read_file: Path = config.MET_OBJECTS_PATH,
        count: int = 1,
        classification: str = config.PAINTING_CLASSIFICATION,
    ) -> List[Tuple[Path, Path]]:
        objects = await self.read_file(read_file)

        # Фильтрация объектов, по классификации, по умолчанию картинка
        random_objects = self.select_objects_sample(objects, count, classification)

        return await self.process_by_object_list(random_objects)
