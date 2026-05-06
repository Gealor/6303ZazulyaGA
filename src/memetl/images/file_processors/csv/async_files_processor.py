import csv
from pathlib import Path

import aiofiles

import memetl.config as config
from memetl.dataclass import MetObject
from memetl.exceptions.files import IncorrectFormatCSVException
from memetl.images.file_processors.abstract_file_processor import (
    AbstractAsyncFileProcessor,
)
from memetl.logger import log


class CSVAsyncFileProcessor(AbstractAsyncFileProcessor):
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


