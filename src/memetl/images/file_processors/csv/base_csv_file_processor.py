import csv
import random
from pathlib import Path

import memetl.config as config
from memetl.dataclass import BaseObject, MetObject
from memetl.images.exceptions import IncorrectFormatCSVException
from memetl.images.file_processors.abstract_file_processor import AbstractFileProcessor
from memetl.logger import log


class BaseCSVFileProcessor(AbstractFileProcessor):
    def read_file(self, file: Path | str = config.MET_OBJECTS_PATH) -> list[MetObject]:
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

    def _select_objects_sample(
        self,
        objects: list[MetObject],
        count: int,
        classification: str = config.PAINTING_CLASSIFICATION,
    ) -> list[MetObject]:
        '''
        Фильтрация и составление выборки объектов, по классификации, по умолчанию картинка
        '''
        log.info("Фильтрация данных...")
        filtered_objects = [
            elem for elem in objects if elem.classification == classification
        ]
        # Выбираю случайный объект
        log.info("Выбор %d случайных элементов...", count)
        random_objects = random.sample(filtered_objects, k=count)
        log.debug(
            "IDs выбранных объектов: %s",
            [random_object.object_id for random_object in random_objects]
        )

        return random_objects

