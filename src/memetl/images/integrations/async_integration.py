import json
from pathlib import Path
from typing import Any

import aiofiles
import aiohttp

import memetl.config as config
from memetl.dataclass import ImageObject
from memetl.logger import log


async def _save_metadata_in_file(
    data: dict,
    path: Path = config.WORK_DIR / config.PAINTINGS_DIR_NAME / config.METADATA_FILE,
) -> None:
    log.debug("Сохраняю метаданные в %s...", path.as_posix())
    try:
        async with aiofiles.open(path, mode="w", encoding="utf-8") as file:
            await file.write(json.dumps(data, indent=4, ensure_ascii=False))
    except Exception:
        log.warning("Произошла ошибка при сохранении метаданных.", exc_info=True)
    else:
        log.debug("Метаданные успешно сохранены.")


async def _make_request(
    info_url: str, client_session: aiohttp.ClientSession
) -> dict[str, Any]:
    log.info("Делаю запрос на %s...", info_url)
    async with client_session.get(url=info_url) as response:
        response.raise_for_status()
        data = await response.json()

    return data


async def make_request_and_save_info(
    value: str,
    metadata_path: Path,
    client_session: aiohttp.ClientSession,
    url: str = config.BASE_URL,
) -> ImageObject:
    info_url = url + value
    data = await _make_request(info_url=info_url, client_session=client_session)

    await _save_metadata_in_file(data=data, path=metadata_path)
    try:
        image_object = ImageObject(
            object_id=data["objectID"],
            primary_image=data["primaryImage"],
        )
    except ValueError as e:
        log.error("Некорректный формат ответа: %s", e)
        raise
    except KeyError as e:
        log.error("Не найден ключ в ответе, полученном из URL: %s", e)
        raise

    log.info("Ответ успешно получен.")
    return image_object


async def download_files(
    object_id: str, path: Path, url: str, client_session: aiohttp.ClientSession
):
    log.debug("Скачиваем файл с %s в директорию %s...", url, path.as_posix())
    async with client_session.get(url=url) as response:
        response.raise_for_status()
        content = await response.read()

    async with aiofiles.open(path, mode="wb") as file:
        await file.write(content)

    log.debug("Файл с ID=%s успешно скачан.", object_id)
