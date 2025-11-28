# AI Fairness Workspace

Quick start (macOS, zsh):

```bash
# create the conda env
conda env create -f environment.yml
# activate
conda activate ai-fairness
# run the training and evaluation script
python scripts/run_all_frameworks.py
```

Notes
- /tests contains basic pytest with minimum coverage to ensure model training and Fairlearn run without error.
- On Apple silicon (M1/M2), prefer installing conda from Miniforge/Miniconda that supports arm64; some packages (TensorFlow) may need special wheels. See troubleshooting below.

Troubleshooting

- If `aif360` fails to install via pip in conda env, try creating a pip-based virtualenv or use conda-forge builds where available.
- For TensorFlow on Apple silicon, see https://developer.apple.com/metal/tensorflow-plugin/ (if you need GPU/accelerator support).
