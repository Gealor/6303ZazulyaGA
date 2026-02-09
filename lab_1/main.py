import csv
import os
import random
from pathlib import Path

from image_processing import process_image
from logger import log
from dataclass import MetObject
from integration import download_files, make_request

random.seed(52)

DIR_NAME = "paintings"
BASE_DIR = Path(__file__).parent

MET_OBJECTS_FILE = BASE_DIR / "MetObjects.csv"
PAINTING_CLASSIFICATION = "Paintings"

ORIGINAL_IMAGE = "original.jpg"

def create_dir(name: str = DIR_NAME) -> Path:
    '''
    Создание нужной директории с именем name
    '''
    painting_dir = BASE_DIR / name
    if not os.path.exists(painting_dir):
        log.info("Создание директории %s...", name)
        os.makedirs(painting_dir)
    else:
        log.info("Директория уже создана. Пропускаем...")
    
    return painting_dir


def read_csv_file(file: Path = MET_OBJECTS_FILE) -> list[MetObject]:
    '''
    Чтение .csv файла и получение всех объектов с их идентификаторами и классификациями(классами)
    '''
    result = []
    log.info("Чтение .csv файла...")
    with open(file, mode="r", encoding="utf-8-sig") as f: # sig, чтобы убрать \ufeff символ
        try:
            csv_reader = csv.DictReader(f)
        except Exception as e:
            log.error("Ошибка при чтении csv файла: %s", e)
            raise e
        for row in csv_reader:
            obj = MetObject(
                object_id=row["Object ID"],
                classification=row["Classification"]
            )
            result.append(obj)
    log.info("Файл прочитан успешно.")
    return result

def save_images(file_to_url: dict[str, str], paintings_dir: Path):
    '''
    Скачивание изображений в заданную директорию paintings_dir.
    '''
    for key, elem in file_to_url.items():
        SAVED_FILE = paintings_dir / key
        download_files(SAVED_FILE, elem)


def main():
    path = create_dir()
    objects_from_csv = read_csv_file()
    
    # Фильтрация объектов, по классификации, чтобы это была картинка
    log.info("Фильтрация данных...")
    objects_from_csv = [elem for elem in objects_from_csv if elem.classification==PAINTING_CLASSIFICATION]
    # Выбираю случайный объект
    log.info("Выбор случайного элемента...")
    random_painting = random.choice(objects_from_csv)
    log.info("Выбран объект с ID = %s", random_painting.object_id)

    # Получаю данные по http запросу
    image_object = make_request(random_painting.object_id)
    # Скачиваю изображения
    file_to_url = {
        ORIGINAL_IMAGE: image_object.primary_image,
    }
    save_images(file_to_url, paintings_dir=path)

    process_image(path, name_original=ORIGINAL_IMAGE)

if __name__ == "__main__":
    main()
