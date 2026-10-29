from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "ground_truth"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Start with screening-style features only. Disease-specific tests are kept
# in the dataset but excluded here to avoid trivial target leakage.
NUMERIC_FEATURES = [
    "fever_duration_days",
    "temperature_c",
    "wbc_count",
    "platelet_count",
    "hemoglobin",
    "neutrophil_percent",
    "lymphocyte_percent",
    "crp_mg_l",
    "procalcitonin_ng_ml",
    "oxygen_saturation_percent",
    "cough_duration_days",
]
TARGET = "target_pattern"

def main():
    candidates = sorted(DATA_DIR.glob("ground_truth_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No ground_truth_*.csv found in {DATA_DIR}. "
            "Run scripts/generate_reports.py first."
        )
    data_path = candidates[-1]
    df = pd.read_csv(data_path)
    required = NUMERIC_FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].astype(str)

    print("Dataset:", data_path.name)
    print("Rows:", len(df))
    print("Screening input features:", len(NUMERIC_FEATURES))
    print("\nSynthetic class distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\nHeld-out test accuracy:", round(accuracy_score(y_test, pred), 4))
    print("\nPer-class report:")
    print(classification_report(y_test, pred, zero_division=0))
    print("Confusion matrix (class order below):")
    print(confusion_matrix(y_test, pred, labels=model.classes_))
    print("Class order:", list(model.classes_))

    output = MODEL_DIR / "synthetic_infection_pattern_model.joblib"
    joblib.dump({"model": model, "features": NUMERIC_FEATURES}, output)
    print("\nSaved model:", output)
    print("CAUTION: performance on generated data does not show real-world diagnostic accuracy.")

if __name__ == "__main__":
    main()
