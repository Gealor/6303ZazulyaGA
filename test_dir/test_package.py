import asyncio

from memetl.analysis import run_full_analysis
from memetl.decorators import time_meter_decorator
from memetl.main import concurency_pipeline_main

if __name__ == '__main__': # ОБЯЗАТЕЛЬНО ПРИ ИСПОЛЬЗОВАНИИ ProcessPoolExecutor (из-за использования spawn для создания процессов)
    asyncio.run(concurency_pipeline_main(count = 5, only_analize = False))
