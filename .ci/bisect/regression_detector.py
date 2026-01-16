from _operator import sub
import argparse
import subprocess

# The default regression threshold is 10%
DEFAULT_REGRESSION_THRESHOLD = 0.1

def get_baseline(baseline_log) -> float:
    with open(baseline_log, "r") as f:
        last_line = f.readlines()[-1]
    return float(last_line.strip())

def get_current_value(stdout_lines) -> float:
    last_line = stdout_lines[-1]
    return float(last_line.strip())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repro", type=str, required=True, help="Command to reproduce the regression")
    parser.add_argument("--baseline", action="store_true", help="Whether this is running the baseline")
    parser.add_argument("--baseline-log", type=str, help="Baseline standard output file")
    parser.add_argument("--functional", type=str, help="This is to bisect a functional regression (e.g., exception, OOM, etc)")
    parser.add_argument("--regression-threshold", type=float, default=DEFAULT_REGRESSION_THRESHOLD, help="Threshold for regression detection")
    args = parser.parse_args()
    if "--simple-output" not in args.repro:
        print("Regression detector requires --simple-output as it only reads the last line in the benchmark output.")
        exit(1)
    if args.baseline_log:
        assert os.path.exists(args.baseline_log), f"Baseline log file {args.baseline_log} does not exist."
    # run the command
    cmdline = args.repro.split()
    if args.functional:
        try:
            subprocess.check_call(cmdline)
        except subprocess.CalledProcessError as e:
            print(f"cmd line {cmdline} failed: {e}")
            exit(e.returncode)
    else:
        baseline = get_baseline(args.baseline_log)
        p = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.STDERR)
        assert p.stdout is not None
        stdout_lines = []
        for line in p.stdout:
            print(line)
            stdout_lines.append(line)
        rc = p.wait()
        if args.baseline:
            with open(args.baseline_log, "w") as f:
                f.write("\n".join(stdout_lines))
            exit(rc)
        # if subprocess failed, exit with the return code
        if rc:
            return rc
        # otherwise, check for the perf regression
        current_value = get_current_value(stdout_lines)
        smaller_value = min(baseline, current_value)
        larger_value = max(baseline, current_value)
        if larger_value > smaller_value * (1 + args.regression_threshold):
            print(f"Regression detected: current value {current_value} regresses over the baseline {baseline} by {args.regression_threshold*100}%)")
            exit(1)
        else:
            print(f"No regression detected: current value {current_value} regresses over the baseline {baseline} by {args.regression_threshold*100}%)")
            exit(0)
