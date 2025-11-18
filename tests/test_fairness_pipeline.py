"""
Basic pytest to run training and a simple metric evaluation.
"""
import subprocess
import sys


def test_train_and_run():
    # Run the training script as a module to ensure package imports resolve
    res = subprocess.run([sys.executable, "-m", "scripts.train_model"], capture_output=True, text=True)
    assert res.returncode == 0, f"Training failed: {res.stderr}"
    # Run Fairlearn check (should not crash)
    res2 = subprocess.run([sys.executable, "-m", "scripts.run_fairlearn_test"], capture_output=True, text=True)
    assert res2.returncode == 0, f"Fairlearn test failed: {res2.stderr}"
