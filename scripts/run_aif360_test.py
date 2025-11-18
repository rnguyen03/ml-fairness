"""
Run a simple AIF360 fairness check (disparate impact) using the trained model.
Note: Requires `aif360` package installed in the environment.
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from scripts.load_adult import load_adult

try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
except Exception as e:
    BinaryLabelDataset = None
    BinaryLabelDatasetMetric = None


def run_aif360_check():
    df = load_adult()
    if "income_binary" not in df.columns:
        raise RuntimeError("income_binary column required")
    X = df.drop(columns=["income_binary"])
    y = df["income_binary"]
    model = joblib.load("models/logreg_adult.joblib")
    preds = model.predict(X)

    # Build aif360 dataset
    if BinaryLabelDataset is None:
        print("aif360 not available in this environment. Install aif360 to run this script.")
        return

    # Build BinaryLabelDataset from DataFrame; use 'sex_Male' one-hot as protected attr if present
    prot_name = "sex_Male" if "sex_Male" in X.columns else ("sex" if "sex" in X.columns else None)
    if prot_name is None:
        # try infer any sex_* column
        candidates = [c for c in X.columns if c.startswith("sex_")]
        if candidates:
            prot_name = candidates[0]
    if prot_name is None:
        print("No protected attribute found for AIF360 demo")
        return

    df_all = pd.concat([X, y.rename("income_binary")], axis=1)
    dataset_true = BinaryLabelDataset(favorable_label=1,
                                      unfavorable_label=0,
                                      df=df_all,
                                      label_names=["income_binary"],
                                      protected_attribute_names=[prot_name])

    df_pred = df_all.copy()
    df_pred["income_binary"] = preds
    dataset_pred = BinaryLabelDataset(favorable_label=1,
                                      unfavorable_label=0,
                                      df=df_pred,
                                      label_names=["income_binary"],
                                      protected_attribute_names=[prot_name])

    # dataset-level fairness
    bl_metric = BinaryLabelDatasetMetric(
        dataset_true,
        privileged_groups=[{prot_name: 1}] if prot_name.endswith("_Male") else [{prot_name: "Male"}],
        unprivileged_groups=[{prot_name: 0}] if prot_name.endswith("_Male") else [{prot_name: "Female"}]
    )

    # classification fairness metrics comparing true vs predicted
    cls_metric = ClassificationMetric(
        dataset_true,
        dataset_pred,
        unprivileged_groups=[{prot_name: 0}] if prot_name.endswith("_Male") else [{prot_name: "Female"}],
        privileged_groups=[{prot_name: 1}] if prot_name.endswith("_Male") else [{prot_name: "Male"}],
    )

    results = {
        "disparate_impact": bl_metric.disparate_impact(),
        "statistical_parity_difference": bl_metric.statistical_parity_difference(),
        "equal_opportunity_difference": cls_metric.equal_opportunity_difference(),
        "average_odds_difference": cls_metric.average_odds_difference()
    }

    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    with open(outdir / "aif360_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("AIF360 metrics saved to outputs/aif360_metrics.json")


if __name__ == "__main__":
    run_aif360_check()
