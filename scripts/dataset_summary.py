import json
from pathlib import Path
import pandas as pd
from scripts.load_adult import load_adult


def main():
    df = load_adult()
    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)

    summary = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns[:50]),
        "label": "income_binary",
        "label_counts": df["income_binary"].value_counts().to_dict() if "income_binary" in df.columns else {},
        "sex_dummy_present": "sex_Male" in df.columns,
    }

    # Save summary JSON
    with open(outdir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # If sex_Male exists, save distribution
    if "sex_Male" in df.columns:
        grp = df.groupby(df["sex_Male"].map({1: "Male", 0: "Female"}))
        grp_size = grp.size().rename("count").reset_index()
        grp_size.to_csv(outdir / "dataset_group_counts.csv", index=False)


if __name__ == "__main__":
    main()
