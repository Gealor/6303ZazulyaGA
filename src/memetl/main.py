import asyncio
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal

import aiohttp

from memetl.analysis.pipeline import analyze_file, run_full_analysis, run_pipeline
from memetl.argparser import prepare_argparser
from memetl.decorators import async_time_meter_decorator, time_meter_decorator
from memetl.images.artwork import ArtworkColorful, ArtworkGrayscale
from memetl.images.file_processors import CSVAsyncFileProcessor, CSVFileProcessor
from memetl.images.handlers import handle_one_image
from memetl.images.image_processors.image_processor import ImageProcessor
from memetl.logger import log

random.seed(52)


def _test_add(saved_file_path: Path, saved_file_dir: Path):
    artwork1 = ArtworkColorful(path=saved_file_path)
    log.info("Тест сложения с выделенными границами...")
    artwork2 = ArtworkGrayscale(img=artwork1.handmade_highlight_borders())
    result = artwork1 + artwork2
    result.save_image(path=saved_file_dir / "original_plus_highlight_borders.jpg")

    log.info("Тест сложения с размытием Гаусса...")
    artwork3 = ArtworkGrayscale(img=artwork1.handmade_gaussian_blur())
    result = artwork1 + artwork3
    result.save_image(path=saved_file_dir / "original_plus_gaussian_blur.jpg")

    log.info("Тест сложения grayscale изображения и выделенные границы")
    artwork_gray = ArtworkGrayscale(path=saved_file_path)
    artwork_sobel = ArtworkGrayscale(img=artwork_gray.handmade_highlight_borders())
    result = artwork_gray + artwork_sobel
    result.save_image(path=saved_file_dir / "grayscale_plus_highlight_borders.jpg")
    result = artwork_sobel + artwork_gray
    result.save_image(path=saved_file_dir / "highlight_borders_plus_grayscale.jpg")


@time_meter_decorator
def analyze_csv(version: Literal["old", "new"]):
    log.info("Запуск '%s' версии аналитики", version)
    if version == "old":
        df_clean = run_pipeline()
        stats_df, timeline_df = analyze_file(df_clean)
        # print(stats_df[:10])
    else:
        run_full_analysis()


@time_meter_decorator
def sync_pipeline_main(count: int, analyze_file: bool = True, only_analize: bool = True):
    if analyze_file:
        log.info("Начало аналитики...")
        analyze_csv("new")
        if only_analize:
            return
        log.info("Данные проанализированны.\n")

    file_processor = CSVFileProcessor()
    log.info("=== Синхронная обработка данных ===")
    log.info("Начало подготовки данных...")
    list_paths = file_processor.start_pipeline(count=count)
    if not list_paths:
        log.info("Нет данных для обработки.")
        return
    log.info("Итого изображений: %d", len(list_paths))

    for saved_file_path, saved_file_dir in list_paths:
        handle_one_image(file_path=saved_file_path, file_dir=saved_file_dir)

    # _test_add(saved_file_path, saved_file_dir)


@async_time_meter_decorator
async def concurency_pipeline_main(
    count: int, analyze_file: bool = True, only_analize: bool = True
):
    if analyze_file:
        log.info("Начало аналитики...")
        analyze_csv("new")
        if only_analize:
            return
        log.info("Данные проанализированны.\n")

    log.info("=== Параллельная обработка данных ===")
    async with aiohttp.ClientSession() as session:
        file_processor = CSVAsyncFileProcessor(client_session=session)
        log.info("Начало подготовки данных...")
        list_paths = await file_processor.start_pipeline(count=count)

    if not list_paths:
        log.info("Нет данных для обработки.")
        return

    log.info("Итого изображений: %d \n", len(list_paths))
    # tasks = [
    #     Process(target=handle_one_image, args=(saved_file_path, saved_file_dir))
    #     for saved_file_path, saved_file_dir in list_paths
    # ]
    # for p in tasks:
    #     p.start()

    # БЛОКИРУЕТ EVENT_LOOP
    # for p in tasks: # ждем завершения ВСЕХ процессов
    #     p.join()

    log.debug("Запуск пула процессов для обработки изображений...")
    loop = asyncio.get_running_loop()

    # Используем ProcessPoolExecutor, т.к по умолчанию использует количество ядер CPU, т.е. не будет проблем с OOM
    with ProcessPoolExecutor() as pool:
        # Создаем задачи для Event Loop, которые будут выполняться в пуле процессов
        processing_tasks = [
            loop.run_in_executor(pool, handle_one_image, saved_file_path, saved_file_dir)
            for saved_file_path, saved_file_dir in list_paths
        ]

        # Асинхронно дожидаемся завершения ВСЕХ тяжелых вычислений.
        # При этом сам Event Loop не блокируется.
        await asyncio.gather(*processing_tasks)


if __name__ == "__main__":
    parser = prepare_argparser()
    args = parser.parse_args()
    if args.parallel:
        asyncio.run(
            concurency_pipeline_main(
                count=args.count,
                analyze_file=args.analyze_file,
                only_analize=args.only_analyze,
            )
        )
    else:
        sync_pipeline_main(
            count=args.count,
            analyze_file=args.analyze_file,
            only_analize=args.only_analyze,
        )
