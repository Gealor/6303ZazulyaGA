import json
from pathlib import Path
from typing import Any

import requests

import memetl.config as config
from memetl.dataclass import ImageObject
from memetl.logger import log


def _save_metadata_in_file(
    data: dict[str, Any] | list[dict[str, Any]],
    path: Path | str = config.WORK_DIR / config.PAINTINGS_DIR_NAME / config.METADATA_FILE,
) -> None:
    log.debug("Сохраняю метаданные в %s...", path.as_posix() if isinstance(path, Path) else path)
    try:
        with open(path, mode="w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
    except Exception:
        log.warning("Произошла ошибка при сохранении метаданных.")
    else:
        log.debug("Метаданные успешно сохранены.")


def make_request(
    value: str,
    metadata_path: Path,
    url: str = config.BASE_URL,
) -> ImageObject:
    info_url = url + value
    log.info("Делаю запрос на %s...", info_url)
    response = requests.get(url=info_url)
    response.raise_for_status()

    data = response.json()
    _save_metadata_in_file(data=data, path=metadata_path)
    try:
        image_object = ImageObject(
            object_id=data.get("objectID"),
            primary_image=data.get("primaryImage") or data.get("primaryImageSmall"),
        )
    except ValueError as e:
        log.error("Некорректный формат ответа: %s", e)
        raise

    log.info("Ответ успешно получен.")
    return image_object


def download_files(path: Path, url: str):
    log.debug("Скачиваем файл с %s в директорию %s...", url, path.as_posix())
    response = requests.get(url)
    with open(path, mode="wb") as file:
        file.write(response.content)

    log.debug("Файл успешно скачан.")
