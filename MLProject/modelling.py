import os
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

import mlflow
import mlflow.sklearn

# ===============================
# PATH DATA
# ===============================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "titanic_preprocessing"
TRAIN_PATH = DATA_DIR / "train_preprocessed.csv"
TEST_PATH = DATA_DIR / "test_preprocessed.csv"

TARGET_COL = "survived"


def main():
    # ===============================
    # DEBUG (penting buat CI)
    # ===============================
    print("=== DEBUG PATH ===")
    print("CWD:", os.getcwd())
    print("BASE_DIR:", BASE_DIR)
    print("DATA_DIR:", DATA_DIR, "exists:", DATA_DIR.exists())
    print("TRAIN_PATH:", TRAIN_PATH, "exists:", TRAIN_PATH.exists())
    print("TEST_PATH :", TEST_PATH, "exists:", TEST_PATH.exists())
    print("==================")

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Train file not found: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_PATH}")

    # ===============================
    # LOAD DATA
    # ===============================
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if TARGET_COL not in train_df.columns:
        raise KeyError(
            f"Target column '{TARGET_COL}' not found in train_df columns: {train_df.columns.tolist()}"
        )
    if TARGET_COL not in test_df.columns:
        raise KeyError(
            f"Target column '{TARGET_COL}' not found in test_df columns: {test_df.columns.tolist()}"
        )

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    # ===============================
    # MLFLOW SETUP
    # ===============================
    # Pakai local file store (aman untuk lokal & CI)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("titanic_baseline_rf")

    # Autolog
    mlflow.sklearn.autolog()

    # ===============================
    # TRAIN & EVAL
    # ===============================
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # (opsional) log metric manual juga, biar jelas di UI
    mlflow.log_metric("accuracy_manual", float(acc))
    mlflow.log_metric("f1_manual", float(f1))

    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    # Kalau jalanin manual (python modelling.py), belum ada active run → buat run
    # Kalau jalan lewat `mlflow run`, biasanya sudah ada active run → jangan buat lagi
    active = mlflow.active_run()
    if active is None:
        with mlflow.start_run(run_name="random_forest_baseline"):
            main()
    else:
        # run sudah dibuat oleh MLflow Project
        main()
