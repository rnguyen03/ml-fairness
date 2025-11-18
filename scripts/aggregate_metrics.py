"""
Aggregate outputs from Fairlearn and AIF360 into a concise Markdown summary for papers.
"""
from pathlib import Path
import json
import pandas as pd


def main():
    outdir = Path("outputs")
    fairlearn_by = outdir / "fairlearn_by_group.csv"
    fairlearn_overall = outdir / "fairlearn_overall.csv"
    fairlearn_fairness = outdir / "fairlearn_fairness.json"
    aif_json = outdir / "aif360_metrics.json"
    # Google outputs (prefer TFMA/WIT if present, else local fallback)
    google_wit_by = outdir / "google_wit_by_group.csv"
    google_tfma_json = outdir / "google_tfma_metrics.json"  # optional placeholder if exported from Colab
    google_local_by = outdir / "google_local_by_group.csv"
    google_local_overall = outdir / "google_local_overall.csv"
    google_local_fairness = outdir / "google_local_fairness.json"

    parts = []

    if fairlearn_by.exists():
        df_by = pd.read_csv(fairlearn_by)
        parts.append("## Fairlearn metrics by group (sex)\n\n" + df_by.to_csv(index=False))
    else:
        parts.append("## Fairlearn metrics by group\n\nNot found.")

    if fairlearn_overall.exists():
        df_overall = pd.read_csv(fairlearn_overall, index_col=0)
        parts.append("\n\n## Fairlearn overall metrics\n\n" + df_overall.to_csv())
    else:
        parts.append("\n\n## Fairlearn overall metrics\n\nNot found.")

    if aif_json.exists():
        data = json.loads(aif_json.read_text())
        # Flatten to two-column table
        df_aif = pd.DataFrame(list(data.items()), columns=["metric", "value"])
        parts.append("\n\n## AIF360 metrics\n\n" + df_aif.to_csv(index=False))
    else:
        parts.append("\n\n## AIF360 metrics\n\nNot found.")

    # Fairlearn fairness measures if available
    if fairlearn_fairness.exists():
        data = json.loads(fairlearn_fairness.read_text())
        df_ff = pd.DataFrame(list(data.items()), columns=["metric", "value"])
        parts.append("\n\n## Fairlearn fairness measures (SPD/DI/EOD/AOD)\n\n" + df_ff.to_csv(index=False))

    # Google metrics section
    if google_wit_by.exists():
        df_gwit = pd.read_csv(google_wit_by)
        parts.append("\n\n## Google What-If (by group)\n\n" + df_gwit.to_csv(index=False))
    elif google_local_by.exists():
        df_glocal = pd.read_csv(google_local_by)
        parts.append("\n\n## Google (local fallback) by group\n\n" + df_glocal.to_csv(index=False))
        if google_local_overall.exists():
            df_gov = pd.read_csv(google_local_overall, index_col=0)
            parts.append("\n\n### Google (local fallback) overall\n\n" + df_gov.to_csv())
        if google_local_fairness.exists():
            data = json.loads(google_local_fairness.read_text())
            df_gfair = pd.DataFrame(list(data.items()), columns=["metric", "value"])
            parts.append("\n\n### Google (local fallback) fairness measures (SPD/DI/EOD/AOD)\n\n" + df_gfair.to_csv(index=False))
    elif google_tfma_json.exists():
        df_gtfma = pd.read_json(google_tfma_json)
        parts.append("\n\n## Google TFMA (summary)\n\n" + df_gtfma.to_csv(index=False))
    else:
        parts.append("\n\n## Google metrics\n\nNot found.")

    summary = "\n\n".join(parts) + "\n"
    # Write main summary and a comparison copy
    (outdir / "summary.md").write_text(summary)
    (outdir / "comparison").mkdir(exist_ok=True)
    (outdir / "comparison" / "summary_all.md").write_text(summary)
    print("Wrote outputs/summary.md and outputs/comparison/summary_all.md")


if __name__ == "__main__":
    main()
