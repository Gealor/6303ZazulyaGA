import time
from typing import Callable, ParamSpec, TypeVar

from logger import log

T = TypeVar("T")
P = ParamSpec("P")


def time_meter_decorator(func: Callable[P, T]) -> Callable[P, T]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start
        log.info("Функция %s выполнилась за %f секунд", func.__name__, execution_time)
        return result

    return wrapper
