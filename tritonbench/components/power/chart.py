import logging
import subprocess
import signal
import torch
import time
import os

# query every 100 ms
QUERY_FREQUENCY = 100
QUERY_STDOUT_FILE = "power.csv"
QUERY_STDERR_FILE = "power.log"
QUERY_COMMAND = """
nvidia-smi -lms {QUERY_FREQUENCY} -i {QUERY_DEVICE} --query-gpu=power.draw.average,power.draw.instant,power.max_limit,temperature.gpu,temperature.memory,clocks.current.sm,clocks.current.memory,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,nounits
"""
POWER_OUTPUT_DIR = None
QUERY_PROC = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _get_cuda_device_id():
    return torch.cuda.current_device()


def power_chart_begin(benchmark_name, output_dir):
    # check no other proc is running 
    assert QUERY_PROC is None, "Power query process must be None to start a new one"
    # clean up the directory
    POWER_OUTPUT_DIR = os.path.join(output_dir, benchmark_name)
    os.mkdir(POWER_OUTPUT_DIR)
    stdout_file_path = os.path.join(POWER_OUTPUT_DIR, QUERY_STDOUT_FILE)
    stderr_file_path = os.path.join(POWER_OUTPUT_DIR, QUERY_STDERR_FILE) 
    # Run the command
    global QUERY_PROC, QUERY_CNT
    query_cmd = QUERY_COMMAND.format(QUERY_FREQUENCY=QUERY_FREQUENCY, QUERY_DEVICE=_get_cuda_device_id()).split(" ")
    with open(stdout_file_path, "w") as stdout_file, open(stderr_file_path, "w") as stderr_file:
        QUERY_PROC = subprocess.Popen(query_cmd, stdout=stdout_file, stderr=stderr_file, start_new_session=True)



def power_chart_end():
    global QUERY_PROC, POWER_OUTPUT_DIR
    assert QUERY_PROC is not None, "Power query process cannot be None"
    # Kill the process
    QUERY_PROC.send_signal(signal.SIGINT)
    time.sleep(0.1)
    assert QUERY_PROC.poll() is not None, "Power query process must be killed to proceed"
    # generate the chart based on csv
    logger.info(f"Power csv saved to {POWER_OUTPUT_DIR}.")
