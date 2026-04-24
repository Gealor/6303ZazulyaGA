import time
from typing import Any, Awaitable, Callable, Coroutine, ParamSpec, TypeVar

from memetl.logger import log

T = TypeVar("T")
P = ParamSpec("P")


def time_meter_decorator(func: Callable[P, T]) -> Callable[P, T]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start
        log.debug("Функция %s выполнилась за %f секунд", func.__name__, execution_time)
        return result

    return wrapper


def async_time_meter_decorator(
    func: Callable[P, Coroutine[Any, Any, T]],
) -> Callable[P, Coroutine[Any, Any, T]]:
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        execution_time = time.perf_counter() - start
        log.debug("Функция %s выполнилась за %f секунд", func.__name__, execution_time)
        return result

    return wrapper
