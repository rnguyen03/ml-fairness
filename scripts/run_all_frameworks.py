"""
Run all three fairness frameworks (Fairlearn, AIF360, Google-local fallback) end-to-end,
then regenerate the consolidated comparison summary.
"""
import subprocess
import sys
from pathlib import Path


def run_step(args):
    print("->", " ".join(args))
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise SystemExit(res.returncode)
    return res


def main():
    # Ensure outputs directory exists
    Path("outputs").mkdir(exist_ok=True)

    # 1) Train model
    run_step([sys.executable, "-m", "scripts.train_model"])  # saves models/logreg_adult.joblib

    # 2) Microsoft Fairlearn
    run_step([sys.executable, "-m", "scripts.run_fairlearn_test"])  # saves outputs/fairlearn_*.csv

    # 3) IBM AIF360
    run_step([sys.executable, "-m", "scripts.run_aif360_test"])  # saves outputs/aif360_metrics.json

    # 4) Google local fallback (no TFMA deps)
    run_step([sys.executable, "-m", "scripts.run_google_local_metrics"])  # saves outputs/google_local_*.csv

    # 5) Aggregate comparison summary
    run_step([sys.executable, "-m", "scripts.aggregate_metrics"])  # writes outputs/comparison/summary_all.md
    print("All frameworks executed. See outputs/comparison/summary_all.md")


if __name__ == "__main__":
    main()
