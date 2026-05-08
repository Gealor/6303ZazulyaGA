import asyncio
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import aiohttp
import click

from memetl import config
from memetl.analysis.pipeline import run_full_analysis
from memetl.decorators import async_time_meter_decorator, time_meter_decorator
from memetl.exceptions.files import FileNotFoundException
from memetl.images.file_processors.csv.async_files_processor import CSVAsyncFileProcessor
from memetl.images.file_processors.json.async_files_processor import (
    JSONAsyncFileProcessor,
)
from memetl.images.file_processors.json.sync_files_processor import JSONFileProcessor
from memetl.images.handlers import handle_one_image
from memetl.images.integrations.async_integration import _save_metadata_in_file
from memetl.logger import log


@async_time_meter_decorator
async def concurency_pipeline(input: Path, output: Path):
    log.info("=== Параллельная обработка данных ===")
    output_folder = output.name
    async with aiohttp.ClientSession() as session:
        file_processor = JSONAsyncFileProcessor(
            client_session=session, save_folder=output_folder
        )
        log.info("Начало подготовки данных...")
        await file_processor._clear_folder()
        await file_processor._create_dir()
        objects = await file_processor.read_file(file=input)
        list_paths = await file_processor.process_by_object_list(objects)

    if not list_paths:
        log.info("Нет данных для обработки.")
        return

    log.info("Итого изображений: %d \n", len(list_paths))

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

@time_meter_decorator
def sync_pipeline(input: Path, output: Path):
    output_folder = output.name
    log.info("=== Синхронная обработка данных ===")
    file_processor = JSONFileProcessor(save_folder=output_folder)
    log.info("Начало подготовки данных...")
    objects = file_processor.read_file(file=input)
    list_paths = file_processor.process_by_object_list(objects)

    if not list_paths:
        log.info("Нет данных для обработки.")
        return

    log.info("Итого изображений: %d \n", len(list_paths))

    for saved_file_path, saved_file_dir in list_paths:
        handle_one_image(file_path=saved_file_path, file_dir=saved_file_dir)


@click.group()
def memetl():
    """
    Консольный интерфейс для чтения MetObjects.csv файла, скачивания информации об изображениях и их обработке.

    Выполнил: Зазуля Георгий Алексеевич
    Группа: 6303-010302D
    """


@memetl.command()
@click.option(
    "--num",
    default=1,
    help="Количество изображений, которое необходимо записать. Выберет случайные num элементов. По умолчанию 1.",
)
@click.option(
    "--data",
    default=config.MET_OBJECTS_PATH,
    type=click.Path(path_type=Path),
    help="Путь до .csv файла, относительно рабочей директории, откуда запускается интерфейс.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Путь до файла, куда необходимо сохранить метаданные об изображениях, относительно рабочей директории, откуда запускается интерфейс.",
)
def prepare(num: int, data: Path, output: Path):
    """
    Подготавливает .json файл с информацией о выбранных изображениях
    """
    if not data.exists():
        raise FileNotFoundException(data)

    if not data.is_absolute():
        data = config.WORK_DIR / data

    if not output.is_absolute():
        output = config.WORK_DIR / output

    async def read_file_and_save_metadata(num: int, data: Path, output: Path):
        if not output.exists():
            output.touch(exist_ok=True)

        async with aiohttp.ClientSession() as session:
            file_processor = CSVAsyncFileProcessor(
                client_session=session,
            )
            objects = await file_processor.read_file(file=data)
            random_objects = file_processor.select_objects_sample(objects, num)
            dict_objects = [asdict(obj) for obj in random_objects]
            await _save_metadata_in_file(data=dict_objects, path=output)

    asyncio.run(read_file_and_save_metadata(num=num, data=data, output=output))


@memetl.command()
@click.option(
    "--input",
    required=True,
    type=click.Path(path_type=Path),
    help=(
        "Путь до .json файла откуда прочитать информацию об изображениях, относительно рабочей директории, откуда запускается интерфейс."
        "Замечание: Нужно подготовить файл с информацией о выбранных изображениях, для этого воспользуйтесь инструментом memetl prepare."
    ),
)
@click.option(
    "--output",
    required=True,
    default=config.PAINTINGS_DIR,
    type=click.Path(path_type=Path),
    help="Путь до папки для скачанных и обработанных файлов, относительно рабочей директории, откуда запускается интерфейс.",
)
@click.option(
    "--parallel",
    is_flag=True,
    default=False,
    help="Включить режим параллельного скачивания и обработки. По умолчанию False.",
)
def process(input: Path, output: Path, parallel: bool):
    """
    Скачивает и обрабатывает изображения
    """
    if not input.exists():
        raise FileNotFoundException(input)

    if not input.is_absolute():
        input = config.WORK_DIR / input
    if not output.is_absolute():
        output = config.WORK_DIR / output

    if parallel:
        asyncio.run(concurency_pipeline(input, output))
    else:
        sync_pipeline(input, output)


@memetl.command()
@click.option(
    "--csv",
    default=config.MET_OBJECTS_PATH,
    type=click.Path(path_type=Path),
    help="Путь до MetObjects.csv файла.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Путь до папки, куда сохранять результаты анализа.",
)
def analyze(csv: Path, output: Path):
    """
    Анализ MetObjects.csv файла согласно варианту:
        Вариант 4. Анализ продолжительности процесса создания объектов
            1. Для топ-10 самых часто встречающихся материалов (Medium) найти среднюю продолжительность процесса создания объекта
        (Object Begin Date и Object End Date), 95% доверительный интервал и 95% интервал рассеяния. Построить столбцовую диаграмму.
            2. Для материала с наибольшим средним сроком создания объекта
        построить график изменения этого срока во времени, со скользящим средним.
    """
    if not csv.exists():
        raise FileNotFoundException(csv)

    if not csv.is_absolute():
        csv = config.WORK_DIR / csv
    if not output.is_absolute():
        output = config.WORK_DIR / output
    run_full_analysis(file_path=csv, output_folder=output)


if __name__ == "__main__":
    memetl()
