
from pathlib import Path
import json
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

TARGET = "target"
LABEL = "label"
DIFFICULTY = "difficulty"

CATEGORICAL = ["protocol_type", "service", "flag"]


def main():
    train_path = DATA_DIR / "train_processed.csv"
    test_path = DATA_DIR / "test_processed.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Processed datasets not found. Run src\\preprocessing.py first."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Exclude the original attack name, difficulty score, and target from features.
    excluded = {TARGET, LABEL, DIFFICULTY}
    feature_columns = [c for c in train_df.columns if c not in excluded]

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET].astype(int)
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET].astype(int)

    categorical_columns = [
        c for c in CATEGORICAL if c in feature_columns
    ]
    numeric_columns = [
        c for c in feature_columns if c not in categorical_columns
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])

    print("Training Random Forest model...")
    start = time.time()
    model.fit(X_train, y_train)
    training_seconds = time.time() - start

    print("Evaluating on the separate test dataset...")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(
            y_test, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_test, predictions, zero_division=0
        )),
        "f1_score": float(f1_score(
            y_test, predictions, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "training_seconds": round(training_seconds, 2),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "feature_count_before_encoding": int(len(feature_columns)),
        "model_name": "Random Forest",
        "random_state": 42,
    }

    model_path = MODEL_DIR / "threat_detection_pipeline.joblib"
    metrics_path = OUTPUT_DIR / "model_metrics.json"
    report_path = OUTPUT_DIR / "classification_report.txt"
    matrix_path = OUTPUT_DIR / "confusion_matrix.csv"

    joblib.dump(model, model_path)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    report = classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Attack"],
        zero_division=0,
    )
    report_path.write_text(report, encoding="utf-8")

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["Actual Normal", "Actual Attack"],
        columns=["Predicted Normal", "Predicted Attack"],
    ).to_csv(matrix_path)

    print("\n=== MODEL EVALUATION ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"Training time: {training_seconds:.2f} seconds")

    print("\nClassification report:")
    print(report)

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved confusion matrix: {matrix_path}")


if __name__ == "__main__":
    main()
