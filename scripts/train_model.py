"""
Train a simple LogisticRegression on the Adult dataset and save the model.
"""
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

from scripts.load_adult import load_adult

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(exist_ok=True)


def train_and_save():
    df = load_adult()
    y = df["income_binary"]
    X = df.drop(columns=["income_binary"]) if "income_binary" in df.columns else df.drop(columns=["income_binary"]) 
    # If load_adult returned encoded features already including income_binary,
    # ensure we separate properly
    if "income_binary" not in df.columns:
        raise RuntimeError("income_binary missing from dataframe")

    X = df.drop(columns=["income_binary"])  # features
    y = df["income_binary"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")
    joblib.dump(clf, MODEL_PATH / "logreg_adult.joblib")
    print("Model saved to models/logreg_adult.joblib")


if __name__ == "__main__":
    train_and_save()
