"""Train a decision tree on the Iris dataset and save an evaluation figure.

Run from the project root:
    python src/train.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write files, do not open a window
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

OUTPUTS = Path("outputs")


def main() -> float:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy: {accuracy:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    OUTPUTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=iris.target_names, ax=ax, colorbar=False)
    ax.set_title("Iris decision tree, confusion matrix")
    fig.tight_layout()
    out = OUTPUTS / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    return accuracy


if __name__ == "__main__":
    main()
