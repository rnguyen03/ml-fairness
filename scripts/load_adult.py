"""
Simple loader/preprocessor for the UCI Adult dataset using sklearn's fetch_openml.
Outputs a DataFrame with a binary label column 'income_binary' (0/1) and example protected attribute 'sex'.
"""
import pandas as pd
from sklearn.datasets import fetch_openml


def load_adult(as_frame=True, sample_frac=None):
    """Return a preprocessed pandas DataFrame for the Adult dataset.

    Returns:
        pd.DataFrame with features and 'income_binary' column (1 if >50K, else 0)
    """
    data = fetch_openml("adult", version=2, as_frame=as_frame)
    df = data.frame.copy()
    # Standardize column names and target
    df = df.rename(columns={"class": "income"})
    # Create binary label
    df["income_binary"] = (df["income"].str.strip() == ">50K").astype(int)
    # Keep a simple protected attribute 'sex'
    df["sex"] = df["sex"].str.strip()
    # Quick numeric encoding for simplicity (not for production use)
    df_numeric = pd.get_dummies(df.drop(columns=["income"]), drop_first=True)
    return df_numeric


if __name__ == "__main__":
    df = load_adult()
    print("Loaded Adult dataset with shape:", df.shape)
    print(df.head())
