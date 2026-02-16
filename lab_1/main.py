import random

import config
from core.files_processing import clear_folder, create_dir, read_csv_file
from core.image_processing import process_image
from core.integration import download_files, make_request
from logger import log

random.seed(52)


def main():
    clear_folder(config.BASE_DIR / config.PAINTINGS_DIR_NAME)
    path = create_dir()
    objects_from_csv = read_csv_file()

    # Фильтрация объектов, по классификации, чтобы это была картинка
    log.info("Фильтрация данных...")
    objects_from_csv = [
        elem
        for elem in objects_from_csv
        if elem.classification == config.PAINTING_CLASSIFICATION
    ]
    # Выбираю случайный объект
    log.info("Выбор случайного элемента...")
    random_painting = random.choice(objects_from_csv)
    log.info("Выбран объект с ID = %s", random_painting.object_id)

    # Получаю данные по http запросу
    image_object = make_request(random_painting.object_id)
    # Скачиваю изображение
    download_files(path=path / config.ORIGINAL_IMAGE, url=image_object.primary_image)

    process_image(path, name_original=config.ORIGINAL_IMAGE)


if __name__ == "__main__":
    main()
