# AI Fairness Workspace

This workspace includes example notebooks and scripts to run fairness tests using:
- IBM AIF360
- Microsoft Fairlearn
- Google tools (What-If Tool / Fairness Indicators / TFMA)

Environment (recommended): Conda + VS Code

Quick start (macOS, zsh):

```bash
# create the conda env
conda env create -f environment.yml
# activate
conda activate ai-fairness
# install notebook kernel for this env
python -m ipykernel install --user --name ai-fairness --display-name "Python (ai-fairness)"
```

Run the demos

- Open `notebooks/aif360_demo.ipynb` in VS Code or JupyterLab and run the cells.
- Or run the CLI scripts in `scripts/` (examples: train_model.py, run_aif360_test.py).

Notes

- On Apple silicon (M1/M2), prefer installing conda from Miniforge/Miniconda that supports arm64; some packages (TensorFlow) may need special wheels. See troubleshooting below.

Troubleshooting

- If `aif360` fails to install via pip in conda env, try creating a pip-based virtualenv or use conda-forge builds where available.
- For TensorFlow on Apple silicon, see https://developer.apple.com/metal/tensorflow-plugin/ (if you need GPU/accelerator support).
