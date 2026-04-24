import logging
from datetime import datetime

import memetl.config as config


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # минимальный уровень для логгера
    config.LOG_DIR_PATH.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(config.LOG_DIR_PATH / "app.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(filename)s:%(lineno)d  %(message)s",
        datefmt=config.DATEFMT,
    ))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        fmt=config.FORMAT,
        datefmt=config.DATEFMT,
    ))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


log = setup_logger(__name__)
