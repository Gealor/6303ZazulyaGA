from pathlib import Path
from typing import Iterator

import pandas as pd

from logger import log


def read_chunks(file_path: Path | str, chunksize: int = 50000) -> Iterator[pd.DataFrame]:
    """
    Генератор для чтения файла по частям.
    """
    usecols =['Medium', 'Object Begin Date', 'Object End Date']

    # Читаем данные как строки, чтобы избежать ошибок типизации при смешанных данных
    chunk_iterator = pd.read_csv(
        file_path,
        usecols=usecols,
        chunksize=chunksize,
    )

    for chunk in chunk_iterator:
        log.info("Получен чанк данных размера: %d", len(chunk))
        yield chunk


def process_chunks(chunks: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """
    Генератор для фильтрации, очистки и трансформации данных.
    """
    for chunk in chunks:
        log.info("Фильтрация чанка...")
        # Удаление строк, где нет информации о материале или датах
        chunk = chunk.dropna(subset=['Medium', 'Object Begin Date', 'Object End Date'])

        # Преобразование даты в числа (если будет строка, то она преобразуется в число), 
        # ошибки (текст, 'B.C.' и т.д.) конвертируются в NaN
        chunk['Object Begin Date'] = pd.to_numeric(chunk['Object Begin Date'], errors='coerce')
        chunk['Object End Date'] = pd.to_numeric(chunk['Object End Date'], errors='coerce')

        chunk = chunk.dropna(subset=['Object Begin Date', 'Object End Date'])

        # Продолжительность создания (Duration)
        chunk['Duration'] = chunk['Object End Date'] - chunk['Object Begin Date']

        # Только логически верные данные, маскирование (продолжительность >= 0)
        chunk = chunk[chunk['Duration'] >= 0]

        if not chunk.empty:
            yield chunk[['Medium', 'Object End Date', 'Duration']]
