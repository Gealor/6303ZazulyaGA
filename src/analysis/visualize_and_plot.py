import matplotlib.pyplot as plt
import pandas as pd

import config
from logger import log


def plot_graphs(
    stats_df: pd.DataFrame,
    timeline_df: pd.DataFrame,
    max_duration_medium: int | str,
):
    log.info("Построение графиков...\n")
    log.info("Построение столбцовой диаграммы...")
    fig, ax = plt.subplots(figsize=(10, 5))

    bar = ax.bar(
        stats_df.index,
        stats_df["Mean"],
        yerr = stats_df["CI_Margin"], # черные маленькие усы на графике столбчатой диаграммы
        color = "blue",
        edgecolor = "black",
        capsize = 5,
        label = "Средняя продолжительность (95% доверительный интервал)",
    )

    ax.errorbar(
        stats_df.index,
        stats_df["Mean"],
        yerr = stats_df["Scatter_Margin"], # красные длинные усы на графике столбчатой диаграммы
        ecolor = "red",
        capsize = 3,
        alpha = 0.3,
        label = "Интервал рассеяния (95% доверительный интервал)"
    )

    plt.title("Топ-10 материалов: время создания")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.savefig(config.BAR_FIG_PATH)
    plt.show()

    log.info("Построение графика зависимости от времени со скользящим окном...")
    plt.figure(figsize=(10, 5))
    plt.scatter(
        timeline_df["Object End Date"],
        timeline_df["Duration"],
        alpha = 0.3,
        color = "gray",
        label = "Средняя продолжительность в году",
    )
    plt.plot(
        timeline_df["Object End Date"],
        timeline_df["Rolling_Mean"],
        color = "red",
        label = "Скользящее среднее",
    )
    plt.title(f'Изменение срока создания во времени для: {max_duration_medium}')
    plt.xlabel("Год (Object End Date)")
    plt.ylabel("Продолжительность (лет)")
    plt.legend()
    plt.tight_layout() # автоматически настраивает отступы между элементами
    plt.savefig(config.SLIDE_WINDOW_PATH)
    plt.show()
