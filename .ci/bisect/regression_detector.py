import argparse
import os
import subprocess

# The default regression threshold is 10%
REGRESSION_THRESHOLD = float(os.environ.get("REGRESSION_THRESHOLD", 0.1))
REPRO_CMDLINE = os.environ.get("REPRO_CMDLINE", None)
FUNCTIONAL = os.environ.get("FUNCTIONAL", False)
BASELINE_LOG = os.environ.get("BASELINE_LOG", None)

def get_baseline(baseline_log) -> float:
    with open(baseline_log, "r") as f:
        last_line = f.readlines()[-1]
    return float(last_line.strip())

def get_current_value(stdout_lines) -> float:
    last_line = stdout_lines[-1]
    return float(last_line.strip())

if __name__ == "__main__":
    if "--simple-output" not in REPRO_CMDLINE:
        print("Regression detector requires --simple-output as it only reads the last line in the benchmark output.")
        exit(1)
    assert REPRO_CMDLINE is not None, "REPRO_CMDLINE is not set."
    cmdline = REPRO_CMDLINE.split()

    # functional regression
    if FUNCTIONAL:
        try:
            subprocess.check_call(cmdline)
        except subprocess.CalledProcessError as e:
            print(f"cmd line {cmdline} failed: {e}")
            exit(e.returncode)

    if os.path.exists(BASELINE_LOG):
        has_baseline = True
    else:
        has_baseline = False
    p = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.STDERR)
    assert p.stdout is not None
    stdout_lines = []
    for line in p.stdout:
        print(line)
        stdout_lines.append(line)
    rc = p.wait()
    if not has_baseline:
        with open(BASELINE_LOG, "w") as f:
            f.write("\n".join(stdout_lines))
        exit(rc)
    # if subprocess failed, exit with the return code
    if not rc == 0:
        exit(rc)
    # otherwise, check for the perf regression
    baseline = get_baseline(BASELINE_LOG)
    current_value = get_current_value(stdout_lines)
    smaller_value = min(baseline, current_value)
    larger_value = max(baseline, current_value)
    if larger_value > smaller_value * (1 + REGRESSION_THRESHOLD):
        print(f"Regression detected: current value {current_value} regresses over the baseline {baseline} by {REGRESSION_THRESHOLD*100}%)")
        exit(1)
    else:
        print(f"No regression detected: current value {current_value} regresses over the baseline {baseline} by {REGRESSION_THRESHOLD*100}%)")
        exit(0)
