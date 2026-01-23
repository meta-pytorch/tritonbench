"""
TritonBench Regression Detector
Used toghether with TritonParse bisector for automatic bisection.

Envs to control the behavior:

- FUNCTIONAL: Detect performance or functional regression.
- REPRO_CMDLINE: The repro command line to run, starting with "python run.py", must have "--simple-output".
- BASELINE_LOG: The baseline log file to compare with.
- REGRESSION_THRESHOLD: The regression threshold, default to 10%.

Example usage:

CONDA_ENV=triton-bisect CONDA_DIR=$HOME/local/uv_venvs/triton-bisect \
REPRO_CMDLINE="python run.py --op ragged_attention --only hstu --metrics tflops --simple-output" \
BASELINE_LOG="$PWD/bisect_logs/baseline.log" REGRESSION_THRESHOLD="0.1" \
tritonparseoss bisect --triton-dir $HOME/triton --test-script $PWD/.ci/bisect/regression_detector.py \
--good 945095b1c72a5aba063925be672d39ef2702fbde --bad ec5b855c132f145d6b695d99069753cd314b5c56

"""

import os
import subprocess
from pathlib import Path

# the default regression threshold is 10%
REGRESSION_THRESHOLD = float(os.environ.get("REGRESSION_THRESHOLD", 10.0)) / 100.0
# functional or performance regression
FUNCTIONAL = bool(int(os.environ["FUNCTIONAL"]))
# repro command line
REPRO_CMDLINE = os.environ.get("REPRO_CMDLINE", None)
# baseline log file
BASELINE_LOG = os.environ.get("BASELINE_LOG", None)

REPO_DIR = Path(__file__).parent.parent.parent


def get_baseline(baseline_log) -> float:
    with open(baseline_log, "r") as f:
        last_line = f.readlines()[-1]
    return float(last_line.strip())


def get_current_value(stdout_lines) -> float:
    last_line = stdout_lines[-1]
    return float(last_line.strip())


if __name__ == "__main__":
    if not FUNCTIONAL and "--simple-output" not in REPRO_CMDLINE:
        print(
            "Performance tests require --simple-output as we will only read the last line in the benchmark output."
        )
        exit(1)
    assert REPRO_CMDLINE is not None, "REPRO_CMDLINE is not set."
    cmdline = REPRO_CMDLINE.split()

    # functional regression
    if FUNCTIONAL:
        try:
            subprocess.check_call(cmdline, cwd=REPO_DIR)
        except subprocess.CalledProcessError as e:
            print(f"cmd line {cmdline} failed: {e}")
            exit(e.returncode)
        exit(0)

    assert BASELINE_LOG and os.path.exists(BASELINE_LOG), (
        f"BASELINE_LOG is not set or to a non-exist location: {BASELINE_LOG}."
    )
    baseline_signal = get_baseline(BASELINE_LOG)
    p = subprocess.Popen(cmdline, cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=None)
    assert p.stdout is not None
    stdout_lines = []
    for line in p.stdout:
        decoded_line = line.decode("utf-8").strip()
        print(decoded_line)
        stdout_lines.append(decoded_line)
    rc = p.wait()
    # if subprocess failed, exit with the return code
    if not rc == 0:
        exit(rc)
    # otherwise, check for the perf regression or accuracy regression
    current_value = get_current_value(stdout_lines)
    if current_value == 0 and "accuracy" in REPRO_CMDLINE:
        print("Accuracy test failed, exit with 1.")
        exit(1)
    smaller_value = min(baseline_signal, current_value)
    larger_value = max(baseline_signal, current_value)
    assert smaller_value > 0, "smaller_value should be positive, got zero."
    ratio = (larger_value - smaller_value) / smaller_value * 100
    if larger_value > smaller_value * (1 + REGRESSION_THRESHOLD):
        print(
            f"Regression detected: current value {current_value}, {larger_value} / {smaller_value} - 1 == {ratio}% , threshold {REGRESSION_THRESHOLD * 100}%)"
        )
        exit(1)
    else:
        print(
            f"No regression detected: current value {current_value}, {larger_value} / {smaller_value} - 1 == {ratio}%, threshold {REGRESSION_THRESHOLD * 100}%)"
        )
        exit(0)
