from memetl.analysis import run_full_analysis
from memetl.decorators import time_meter_decorator

run_full_analysis = time_meter_decorator(run_full_analysis)

run_full_analysis()