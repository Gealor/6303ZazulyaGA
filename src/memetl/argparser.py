import argparse
from pathlib import Path

from memetl import config


def prepare_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Скачивание изображений из Met API")
    parser.add_argument(
        "count",
        type=int,
        default=1,
        help="Количество изображений для скачивания (по умолчанию: 1)",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=config.MET_OBJECTS_PATH,
        help="Путь до MetObjects.csv",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Обрабатывать скачивание и обработку файлов параллельно/конкурентно",
    )
    parser.add_argument(
        "--analyze-file", action="store_true", help="Включить аналитику файла"
    )
    parser.add_argument(
        "--only-analyze", action="store_true", help="Только анализ, без скачивания"
    )

    return parser
