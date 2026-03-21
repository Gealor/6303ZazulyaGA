from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from analysis.generators import process_chunks, read_chunks
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

    stats = []
    for medium in top_10_mediums:
        log.info("Получение статистик для %s...", medium)
        med_data = df_top10[df_top10["Medium"]==medium]["Duration"]
        n = len(med_data)
        mean = med_data.mean()
        std = med_data.std() if n > 1 else 0

        # 95% доверительный интервал (квантиль 0.95 ~ 1.96)
        ci_margin = 1.96 * (std / np.sqrt(n))

        # 95% интервал рассеяния
        scatter_margin = 1.96 * std

        data = {
                "Medium": medium,
                "Mean": mean,
                "Std": std,
                "CI_Margin": ci_margin,
                "Scatter_Margin": scatter_margin,
            }
        log.info("Получены данные о материале: %s", data)
        stats.append(data)

    stats_df = pd.DataFrame(stats).set_index("Medium")

    # График изменения времени со скользящим средним
    max_duration_medium = stats_df["Mean"].idxmax() # возвращается индекс материала (его название)
    log.info("Материал с наибольшим средним сроком: %s", max_duration_medium)

    df_max_med = df[df["Medium"] == max_duration_medium].copy()

    # Группировка по году завершения и нахожу среднее по продолжительности за каждый год
    df_timeline = df_max_med.groupby("Object End Date")["Duration"].mean().reset_index()
    df_timeline = df_timeline.sort_values("Object End Date")

    window_size = size_window if len(df_timeline) > size_window else len(df_timeline) // 2

    df_timeline["Rolling_Mean"] = df_timeline["Duration"].rolling(window_size, min_periods=1).mean()

    plot_graphs(stats_df=stats_df, timeline_df=df_timeline, max_duration_medium=max_duration_medium)

    return stats_df, df_timeline




