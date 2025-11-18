"""
Run a simple Fairlearn evaluation using metric_frame.
"""
import joblib
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
from pathlib import Path

from scripts.load_adult import load_adult


def run_fairlearn_check():
    df = load_adult()
    # features/labels
    X = df.drop(columns=["income_binary"]) if "income_binary" in df.columns else df
    y = df["income_binary"]
    model = joblib.load("models/logreg_adult.joblib")
    preds = model.predict(X)

    # Choose a simple sensitive feature if available
    sensitive = None
    if "sex_Male" in X.columns:
        s = X["sex_Male"]
        # Map boolean or 0/1 encoding to human-readable labels
        if s.dtype == bool or set(s.dropna().unique()).issubset({True, False}):
            sensitive = s.map({True: "Male", False: "Female"})
        else:
            sensitive = s.map({1: "Male", 0: "Female"})
    elif "sex" in df.columns:
        sensitive = df["sex"]
    else:
        # fallback: pick the first one-hot sex_* column if present
        sex_cols = [c for c in X.columns if c.startswith("sex_")]
        if sex_cols:
            c = sex_cols[0]
            grp = c.split("sex_")[1]
            sensitive = X[c].map({1: grp, 0: f"not_{grp}"})

    if sensitive is None:
        print("No sensitive attribute found for Fairlearn demo")
        return

    mf = MetricFrame(metrics={
        "selection_rate": selection_rate,
        "tpr": true_positive_rate,
        "fpr": false_positive_rate
    },
    y_true=y,
    y_pred=preds,
    sensitive_features=sensitive)
    # Ensure outputs directory exists and save results
    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    mf.by_group.to_csv(outdir / "fairlearn_by_group.csv")
    # overall is a Series of global metrics; save with a value column for clarity
    pd.Series(mf.overall).to_frame("value").to_csv(outdir / "fairlearn_overall.csv")
    print("By-group metrics saved to outputs/fairlearn_by_group.csv")
    print("Overall metrics saved to outputs/fairlearn_overall.csv")
    # Derive fairness measures: SPD, DI, EOD, AOD
    by = mf.by_group
    try:
        sr = by["selection_rate"]
        tpr = by["tpr"]
        fpr = by["fpr"]

        groups = list(by.index)
        # Prefer Female as unprivileged and Male as privileged when available
        unpriv = "Female" if "Female" in groups else sr.idxmin()
        priv = "Male" if "Male" in groups else sr.idxmax()

        spd = float(sr.loc[unpriv] - sr.loc[priv])
        di = float(sr.loc[unpriv] / sr.loc[priv]) if sr.loc[priv] > 0 else None
        eod = float(tpr.loc[unpriv] - tpr.loc[priv])
        aod = float(0.5 * ((fpr.loc[unpriv] - fpr.loc[priv]) + (tpr.loc[unpriv] - tpr.loc[priv])))

        import json
        fairness = {
            "unprivileged_group": unpriv,
            "privileged_group": priv,
            "statistical_parity_difference": spd,
            "disparate_impact": di,
            "equal_opportunity_difference": eod,
            "average_odds_difference": aod,
        }
        (outdir / "fairlearn_fairness.json").write_text(json.dumps(fairness, indent=2))
        print("Fairlearn fairness saved to outputs/fairlearn_fairness.json")
    except Exception as e:
        print(f"Could not compute Fairlearn fairness measures: {e}")

if __name__ == "__main__":
    run_fairlearn_check()
