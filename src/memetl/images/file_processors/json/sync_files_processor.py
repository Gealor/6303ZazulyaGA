import json
from pathlib import Path

import memetl.config as config
from memetl.dataclass import MetObject
from memetl.exceptions.files import IncorrectFormatJSONException
from memetl.images.file_processors.abstract_file_processor import AbstractFileProcessor
from memetl.logger import log


class JSONFileProcessor(AbstractFileProcessor):
    def read_file(self, file: Path | str = config.MET_OBJECTS_PATH) -> list[MetObject]:
        """
        Чтение .json файла и получение всех объектов с их идентификаторами и классификациями(классами)
        """
        result = []
        log.info("Чтение .json файла...")
        with open(file, mode="r", encoding="utf-8") as f:
            data = json.load(f)

            for row in data:
                try:
                    obj = MetObject(
                        object_id=row["object_id"],
                        classification=row["classification"],
                    )
                    result.append(obj)
                except KeyError as e:
                    log.warning("Ошибка при доступе к аттрибуту: %s", e)
                    raise IncorrectFormatJSONException from KeyError
        log.info("Файл прочитан успешно.")
        return result


