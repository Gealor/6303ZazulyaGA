import csv
from pathlib import Path

import memetl.config as config
from memetl.dataclass import MetObject
from memetl.exceptions.files import IncorrectFormatCSVException
from memetl.images.file_processors.abstract_file_processor import AbstractFileProcessor
from memetl.logger import log


class CSVFileProcessor(AbstractFileProcessor):
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

