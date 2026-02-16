import csv
import os
from pathlib import Path
import shutil

import config
from dataclass import MetObject
from logger import log


def clear_folder(path: Path):
    if path.exists():
        log.info("Удаление папки %s...", path.as_posix())
        shutil.rmtree(path)


def create_dir(name: str = config.PAINTINGS_DIR_NAME) -> Path:
    """
    Создание нужной директории с именем name
    """
    painting_dir = config.BASE_DIR / name
    if not os.path.exists(painting_dir):
        log.info("Создание директории %s...", name)
        os.makedirs(painting_dir)
    else:
        log.info("Директория уже создана. Пропускаем...")

    return painting_dir


def read_csv_file(file: Path = config.MET_OBJECTS_FILE) -> list[MetObject]:
    """
    Чтение .csv файла и получение всех объектов с их идентификаторами и классификациями(классами)
    """
    result = []
    log.info("Чтение .csv файла...")
    with open(
        file, mode="r", encoding="utf-8-sig",
    ) as f:  # sig, чтобы убрать \ufeff символ
        try:
            csv_reader = csv.DictReader(f)
        except Exception as e:
            log.error("Ошибка при чтении csv файла: %s", e)
            raise e

        for row in csv_reader:
            obj = MetObject(
                object_id=row["Object ID"], classification=row["Classification"],
            )
            result.append(obj)

    log.info("Файл прочитан успешно.")
    return result
