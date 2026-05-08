import json
from pathlib import Path

import aiofiles

import memetl.config as config
from memetl.dataclass import MetObject
from memetl.exceptions.files import IncorrectFormatJSONException
from memetl.images.file_processors.abstract_file_processor import (
    AbstractAsyncFileProcessor,
)
from memetl.logger import log


class JSONAsyncFileProcessor(AbstractAsyncFileProcessor):
    async def read_file(
        self, file: Path | str = config.MET_OBJECTS_PATH
    ) -> list[MetObject]:
        """
        Чтение .json файла и получение всех объектов с их идентификаторами и классификациями(классами)
        """
        result = []
        log.info("Чтение .json файла...")
        try:
            async with aiofiles.open(file, mode="r", encoding="utf-8") as f:
                content = await f.read()
        except Exception as e:
            log.error("Ошибка при чтении json файла: %s", e)
            raise

        try:
            data = json.loads(content)
        except Exception as e:
            log.error("Ошибка при парсинге json: %s", e)
            raise

        for row in data:
            try:
                obj = MetObject(
                    object_id=row["object_id"],
                    classification=row["classification"],
                )
                result.append(obj)
            except KeyError as e:
                log.warning("Ошибка при доступе к аттрибуту: %s", e)
                raise IncorrectFormatJSONException from e

        log.info("Файл прочитан успешно.")
        return result
