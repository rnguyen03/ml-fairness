"""
Local fallback for Google fairness metrics (no TFMA/WIT deps).
Computes selection rate, TPR, FPR by group (sex) and overall.
This is provided to compare with Fairlearn and AIF360 when TFMA isn't runnable locally.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from scripts.load_adult import load_adult


def _rates(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pos = y_pred == 1
    neg = y_pred == 0
    tp = int(np.sum((y_true == 1) & pos))
    fp = int(np.sum((y_true == 0) & pos))
    fn = int(np.sum((y_true == 1) & neg))
    tn = int(np.sum((y_true == 0) & neg))
    selection = float(np.mean(pos)) if len(y_true) else 0.0
    tpr = tp / (tp + fn + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    return selection, tpr, fpr


def main():
    out = Path("outputs")
    out.mkdir(exist_ok=True)

    df = load_adult()
    X = df.drop(columns=["income_binary"]) if "income_binary" in df.columns else df
    y = df["income_binary"].astype(int)

    model = joblib.load("models/logreg_adult.joblib")
    probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
    y_pred = (probs >= 0.5).astype(int)

    # sensitive attribute mapping for sex
    if "sex_Male" in X.columns:
        s = X["sex_Male"]
        if s.dtype == bool or set(s.dropna().unique()).issubset({True, False}):
            sens = s.map({True: "Male", False: "Female"})
        else:
            sens = s.map({1: "Male", 0: "Female"})
    else:
        cols = [c for c in X.columns if c.startswith("sex_")]
        if cols:
            grp = cols[0].split("sex_")[1]
            sens = X[cols[0]].map({1: grp, 0: f"not_{grp}", True: grp, False: f"not_{grp}"})
        else:
            sens = pd.Series(["unknown"] * len(X))

    rows = []
    for g, idx in sens.groupby(sens).groups.items():
        sel, tpr, fpr = _rates(y.loc[idx], y_pred[idx])
        rows.append({"group": g, "selection_rate": sel, "tpr": tpr, "fpr": fpr})

    pd.DataFrame(rows).to_csv(out / "google_local_by_group.csv", index=False)

    # overall
    sel_o, tpr_o, fpr_o = _rates(y, y_pred)
    pd.Series({"selection_rate": sel_o, "tpr": tpr_o, "fpr": fpr_o}).to_frame("value").to_csv(out / "google_local_overall.csv")

    print("Saved Google local fallback metrics to outputs/google_local_by_group.csv and outputs/google_local_overall.csv")

    # Derive fairness measures (SPD, DI, EOD, AOD) similar to AIF360 definitions
    try:
        df_by = pd.DataFrame(rows)
        # Choose unprivileged/privileged groups
        groups = list(df_by["group"].values)
        unpriv = "Female" if "Female" in groups else df_by.loc[df_by["selection_rate"].idxmin(), "group"]
        priv = "Male" if "Male" in groups else df_by.loc[df_by["selection_rate"].idxmax(), "group"]

        sr_u = float(df_by.loc[df_by["group"] == unpriv, "selection_rate"].iloc[0])
        sr_p = float(df_by.loc[df_by["group"] == priv, "selection_rate"].iloc[0])
        tpr_u = float(df_by.loc[df_by["group"] == unpriv, "tpr"].iloc[0])
        tpr_p = float(df_by.loc[df_by["group"] == priv, "tpr"].iloc[0])
        fpr_u = float(df_by.loc[df_by["group"] == unpriv, "fpr"].iloc[0])
        fpr_p = float(df_by.loc[df_by["group"] == priv, "fpr"].iloc[0])

        spd = sr_u - sr_p
        di = (sr_u / sr_p) if sr_p > 0 else None
        eod = tpr_u - tpr_p
        aod = 0.5 * ((fpr_u - fpr_p) + (tpr_u - tpr_p))

        fairness = {
            "unprivileged_group": unpriv,
            "privileged_group": priv,
            "statistical_parity_difference": float(spd),
            "disparate_impact": float(di) if di is not None else None,
            "equal_opportunity_difference": float(eod),
            "average_odds_difference": float(aod),
        }
        (out / "google_local_fairness.json").write_text(json.dumps(fairness, indent=2))
        print("Saved Google local fairness metrics to outputs/google_local_fairness.json")
    except Exception as e:
        print(f"Could not compute Google local fairness measures: {e}")


if __name__ == "__main__":
    main()
