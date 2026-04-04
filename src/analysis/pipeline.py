from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from analysis.generators import process_chunks, read_chunks, summarize_chunks
from analysis.visualize_and_plot import plot_graphs
from config import MET_OBJECTS_PATH
from logger import log


def run_pipeline(
    file_path: Path | str = MET_OBJECTS_PATH,
    chunksize: int = 5000,
) -> pd.DataFrame:
    log.info("Начало обработки файла...")
    chunks = read_chunks(file_path, chunksize=chunksize)
    processed = process_chunks(chunks)

    df_clean = pd.concat(processed, ignore_index=True)
    log.info("Обработка завершена. Получено %d записей.", len(df_clean))
    return df_clean

# TODO: вынести это в генератор, обрабатывать чанками, вычисляем промежуточные значения и потом их суммируем к результирующему датафрейму
# переименовать эту функцию в analyze_chunk, внутрь принимать результирующий датафрейм и чанк, для которого будем вычилсять статистику,
# вынести метод plot_graphs в run_pipeline
# и создать отдельный генератор, который для каждого чанка будет применять analyze_chunk.
def analyze_file(
    df: pd.DataFrame,
    top_n: int = 10,
    size_window: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Вариант 4. Вычисление статистик для ТОП-10 материалов.")

    # value_counts считает количество каждого элемента, элементы становятся индексами, а их количество - значениями
    top_10_mediums = df["Medium"].value_counts().nlargest(top_n).index
    # маскирование и получение тех материалов, которые достали выше
    df_top10 = df[df["Medium"].isin(top_10_mediums)]

    stats_df = df_top10.groupby(["Medium"])["Duration"].aggregate(["mean", "std", "count"])

    stats_df["CI_Margin"] = 1.96 * (stats_df["std"] / np.sqrt(stats_df["count"]))
    stats_df["Scatter_Margin"] = 1.96 * stats_df["std"]

    # График изменения времени со скользящим средним
    max_duration_medium = stats_df["mean"].idxmax() # возвращается индекс материала (его название)
    log.info("Материал с наибольшим средним сроком: %s", max_duration_medium)

    df_max_med = df[df["Medium"] == max_duration_medium].copy()

    # Группировка по году завершения и нахожу среднее по продолжительности за каждый год
    df_timeline = df_max_med.groupby("Object End Date")["Duration"].mean().reset_index()
    df_timeline = df_timeline.sort_values("Object End Date")

    window_size = size_window if len(df_timeline) > size_window else len(df_timeline) // 2

    df_timeline["Rolling_Mean"] = df_timeline["Duration"].rolling(window_size, min_periods=1).mean()

    plot_graphs(stats_df=stats_df, timeline_df=df_timeline, max_duration_medium=max_duration_medium)

    return stats_df, df_timeline


def run_full_analysis(
    file_path: Path | str = MET_OBJECTS_PATH,
    top_n: int = 10,
    size_window: int = 20
):
    log.info("Запуск пайплайна обработки...")

    raw_chunks = read_chunks(file_path)
    clean_chunks = process_chunks(raw_chunks)

    res = summarize_chunks(clean_chunks)
    df_stats = res["stats"]
    df_timeline = res["timeline"]

    n = df_stats["count"]
    mean = df_stats["sum_x"] / n
    var = (df_stats["sum_x2"] / n) - (mean**2)
    std = np.sqrt(var.clip(lower=0))

    # Итоговый stats_df
    # Индексы будут Medium, т.к. n, mean и std - это pd.Series
    stats_df = pd.DataFrame(
        {
            'mean': mean,
            'std': std,
            'count': n,
        }
    )

    # Топ-10
    stats_df = stats_df.nlargest(top_n, 'count').copy()
    stats_df["CI_Margin"] = 1.96 * (stats_df["std"] / np.sqrt(stats_df["count"]))
    stats_df["Scatter_Margin"] = 1.96 * stats_df["std"]

    # Таймлайн
    max_duration_medium = stats_df["mean"].idxmax()
    log.info("Материал с наибольшим средним сроком: %s", max_duration_medium)

    # Фильтр таймлайна только для лидера
    leader_timeline = df_timeline[df_timeline["Medium"] == max_duration_medium].copy()

    leader_timeline["Duration"] = leader_timeline["sum_val"] / leader_timeline["count_val"]
    leader_timeline = leader_timeline.sort_values("Object End Date")

    leader_timeline["Rolling_Mean"] = leader_timeline["Duration"].rolling(size_window, min_periods=1).mean()
    plot_graphs(stats_df, leader_timeline, max_duration_medium)


