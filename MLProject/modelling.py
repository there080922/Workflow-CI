import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

import mlflow
import mlflow.sklearn

# ===============================
# PATH DATA
# ===============================
DATA_DIR = "titanic_preprocessing"
TRAIN_PATH = os.path.join(DATA_DIR, "train_preprocessed.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_preprocessed.csv")

TARGET_COL = "survived"

def main():
    # Load dataset hasil preprocessing
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    # Setup MLflow (local)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("titanic_baseline_rf")

    # Autolog 
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="random_forest_baseline"):
        model = RandomForestClassifier(
            n_estimators=200, 
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Accuracy:", acc)
        print("F1 Score:", f1)
        print("\nClassification Report:\n")
        print(classification_report(y_test, y_pred))

        run = mlflow.active_run()
        with open("run._id.txt", "w") as f:
            f.write(run.info.run_id)
        print("Saved run_id:", run.info.run_id)


if __name__ == "__main__":
    main()