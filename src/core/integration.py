from pathlib import Path

import requests

import config
from dataclass import ImageObject
from logger import log


def make_request(value: str, url: str = config.BASE_URL) -> ImageObject:
    info_url = url + value
    log.info(f"Делаю запрос на {info_url}...")
    response = requests.get(url=info_url)
    response.raise_for_status()

    data = response.json()
    try:
        image_object = ImageObject(
            object_id=data.get("objectID"),
            primary_image=data.get("primaryImage"),
        )
    except ValueError as e:
        log.error("Некорректный формат ответа: %s", e)
        raise e

    log.info("Ответ успешно получен.")
    return image_object


def download_files(path: Path, url: str):
    log.info(f"Скачиваем файл с {url} в директорию {path.as_posix()}...")
    response = requests.get(url)
    with open(path, mode="wb") as file:
        file.write(response.content)

    log.info("Файл успешно скачан.")
