import time
from typing import Callable

from logger import log


def time_meter_decorator(func: Callable):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start
        log.info("Функция %s выполнилась за %f секунд", func.__name__, execution_time)
        return result

    return wrapper
