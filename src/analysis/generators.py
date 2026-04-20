from pathlib import Path
from typing import Any, Dict, Iterator

import numpy as np
import pandas as pd

from logger import log


def read_chunks(file_path: Path | str, chunksize: int = 50000) -> Iterator[pd.DataFrame]:
    """
    Генератор для чтения файла по частям.
    """
    usecols = ["Medium", "Object Begin Date", "Object End Date"]

    # Читаем данные как строки, чтобы избежать ошибок типизации при смешанных данных
    chunk_iterator = pd.read_csv(
        file_path,
        usecols=usecols,
        chunksize=chunksize,
    )

    for chunk in chunk_iterator:
        yield chunk


def process_chunks(chunks: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """
    Генератор для фильтрации, очистки и трансформации данных.
    """
    for chunk in chunks:
        # Удаление строк, где нет информации о материале или датах
        chunk = chunk.dropna(subset=["Medium", "Object Begin Date", "Object End Date"])

        # Преобразование даты в числа (если будет строка, то она преобразуется в число),
        # ошибки (текст, 'B.C.' и т.д.) конвертируются в NaN
        chunk["Object Begin Date"] = pd.to_numeric(
            chunk["Object Begin Date"], errors="coerce"
        )
        chunk["Object End Date"] = pd.to_numeric(
            chunk["Object End Date"], errors="coerce"
        )

        chunk = chunk.dropna(subset=["Object Begin Date", "Object End Date"])

        # Продолжительность создания (Duration)
        chunk["Duration"] = chunk["Object End Date"] - chunk["Object Begin Date"]

        # Только логически верные данные, маскирование (продолжительность >= 0)
        chunk = chunk[chunk["Duration"] >= 0]

        if not chunk.empty:
            yield chunk[["Medium", "Object End Date", "Duration"]]


def summarize_chunks(chunks: Iterator[pd.DataFrame]) -> Dict[str, Any]:
    """
    Генератор, который поглощает чанки и накапливает промежуточные статистики для расчета Mean, Std и Timeline.
    """
    # Накопитель для общей статистики по материалам
    full_stats = pd.DataFrame()

    # Накопитель для таймлайна (группировка по материалу и году)
    full_timeline = pd.DataFrame()

    for chunk in chunks:
        # Векторно возвожу в квадрат до группировки
        chunk["Duration_sq"] = chunk["Duration"] ** 2
        # Основная статистика (Mean, Std)
        # agg и aggregate - одно и то же, просто agg - это элиас
        # Medium теперь индекс, т.к. мы группируем по нему
        c_stats = chunk.groupby("Medium").agg(
            sum_x=("Duration", "sum"),
            sum_x2=("Duration_sq", "sum"),
            count=("Duration", "count"),
        )
        if full_stats.empty:
            full_stats = c_stats
        else:
            full_stats = full_stats.add(c_stats, fill_value=0)

        # Таймлайн
        c_timeline = chunk.groupby(["Medium", "Object End Date"])["Duration"].agg(
            sum_val="sum", count_val="count"
        )
        if full_timeline.empty:
            full_timeline = c_timeline
        else:
            full_timeline = full_timeline.add(c_timeline, fill_value=0)

    return {
        "stats": full_stats,
        "timeline": full_timeline.reset_index(),
    }
