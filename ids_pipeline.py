# CYSE 499 – Intrusion Detection System (IDS) Final Project

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV


# CONFIGURATION — Machine Learning CVE

# CIC-IDS2017 CSV files:
DATA_FOLDER = Path(os.getenv("IDS_DATA_PATH", Path(__file__).parent / "data"))
# Limit rows per file as dataset is too large
MAX_ROWS_PER_FILE = 200000

RANDOM_STATE = 42
TEST_SIZE = 0.30

# Output folder for saved plots/models
OUTPUT_DIR = Path(__file__).parent / "Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# HELPER FUNCTIONS

def load_csv_folder(folder_path: Path, max_rows=None) -> pd.DataFrame:
    # Loads ALL .csv files in a folder and concatenates them into one DataFrame.
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"[ERROR] Folder not found: {folder_path}")

    dfs = []
    csv_files = [f for f in folder_path.iterdir() if f.suffix.lower() == ".csv"]

    if not csv_files:
        raise ValueError(f"[ERROR] No CSV files found in: {folder_path}")

    print("\n[INFO] Found CSV files:")
    for f in csv_files:
        print("   -", f.name)

    for csv_path in csv_files:
        print(f"[INFO] Loading {csv_path} ...")
        df = pd.read_csv(csv_path, nrows=max_rows)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Combined dataset shape: {combined.shape}")
    return combined


def clean_dataset(df: pd.DataFrame, label_col: str = "Label") -> pd.DataFrame:
    print("\n[INFO] Cleaning dataset...")

    df = df.dropna(subset=[label_col])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()

    if constant_cols:
        print(f"[INFO] Dropping constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)

    print(f"[INFO] Cleaned dataset shape: {df.shape}")
    return df


def encode_labels(df: pd.DataFrame, label_col="Label", target_col="y"):
    # Encode BENIGN = 0, ATTACK = 1.
    print("\n[INFO] Encoding labels...")

    df[target_col] = df[label_col].apply(
        lambda x: 0 if str(x).upper() == "BENIGN" else 1
    )

    print(df[target_col].value_counts())
    return df


def split_features(df: pd.DataFrame, target_col="y", label_col="Label"):
    # Select numeric features for model training.
    features = df.drop(columns=[target_col, label_col], errors="ignore")
    numeric_cols = features.select_dtypes(include=[np.number]).columns

    X = features[numeric_cols]
    y = df[target_col].astype(int)

    print(f"[INFO] Using {len(numeric_cols)} numeric features.")
    return X, y


def evaluate_and_plot(model, Xtest, ytest, name: str, out_dir: Path) -> float:
    # Evaluates a model and saves confusion matrix + ROC curve plots.
    y_pred = model.predict(Xtest)
    y_proba = model.predict_proba(Xtest)[:, 1]

    print(f"Model Evaluation: {name}")
    print(classification_report(ytest, y_pred, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(ytest, y_pred)
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_confusion_matrix.png", dpi=300)
    plt.close()

    # ROC Curve
    auc = roc_auc_score(ytest, y_proba)
    RocCurveDisplay.from_predictions(ytest, y_proba)
    plt.title(f"{name} - ROC Curve (AUC = {auc:.4f})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_roc_curve.png", dpi=300)
    plt.close()

    return auc


def tune_random_forest(X_train, y_train):
    print("\n[INFO] Running Random Forest Hyperparameter Tuning...")

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )),
    ])

    param_grid = {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [10, 20, 30, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", 0.5],
        "model__bootstrap": [True, False],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=20,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    search.fit(X_train, y_train)

    print("\n[INFO] Best RF AUC:", search.best_score_)
    print("[INFO] Best RF Hyperparameters:")
    for p, v in search.best_params_.items():
        print(f"   {p}: {v}")

    return search.best_estimator_


# MAIN FUNCTION
def main():

    print("   CYSE 499 IDS Pipeline\n")

    # Load CSVs
    df = load_csv_folder(DATA_FOLDER, max_rows=MAX_ROWS_PER_FILE)
    # strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    MAX_TOTAL_ROWS = 200000
    if len(df) > MAX_TOTAL_ROWS:
        print(f"[INFO] Sampling down to {MAX_TOTAL_ROWS} rows...")
        df = df.sample(MAX_TOTAL_ROWS, random_state=42)
    # Clean dataset
    df = clean_dataset(df, label_col="Label")
    # Encode labels
    df = encode_labels(df, label_col="Label", target_col="y")
    # Feature/target split
    X, y = split_features(df, target_col="y", label_col="Label")
    #  Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("[INFO] Starting Random Forest hyperparameter tuning...")
    # Tune Hyperparameters
    best_rf = tune_random_forest(X_train, y_train)
    # Find best Random Forest Model
    rf_auc = evaluate_and_plot(best_rf, X_test, y_test, "RandomForest_Tuned", OUTPUT_DIR)
    # Build logistic regression model and untuned random forest
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                n_jobs=-1
            )),
        ]),
        "RandomForest_Untuned": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )),
        ]),
    }
    # Train and evaluate
    best_model = best_rf
    best_label = "RandomForest_Tuned"
    best_auc = rf_auc

    for name, model in models.items():
        print(f"\n[INFO] Training model: {name}")
        model.fit(X_train, y_train)
        auc = evaluate_and_plot(model, X_test, y_test, name, OUTPUT_DIR)

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_label = name

    # Save Best Model
    model_path = OUTPUT_DIR / "best_ids_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"\n[INFO] BEST MODEL: {best_label} (AUC = {best_auc:.4f})")
    print(f"[INFO] Saved best model to: {model_path}")


if __name__ == "__main__":
    main()
