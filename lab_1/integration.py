from pathlib import Path
import requests

from dataclass import ImageObject
from logger import log


BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1/objects/"


def make_request(value: str, url: str = BASE_URL) -> ImageObject:
    INFO_URL = url + value
    log.info(f"Делаю запрос на {INFO_URL}...")
    response = requests.get(url=INFO_URL)
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


def download_files(dir_image: Path, url: str):
    log.info(f"Скачиваем файл с {url} в директорию {dir_image.as_posix()}...")
    response = requests.get(url)
    with open(dir_image, mode="wb") as file:
        file.write(response.content)
    log.info("Файл успешно скачан.")
