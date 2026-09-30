"""
Model Training Script for Iris Classification
----------------------------------------------
This script loads the Iris dataset from scikit-learn, trains a Random Forest Classifier,
evaluates its performance, and exports:
1. `iris_model.pkl`: The trained scikit-learn model
2. `iris_data.csv`: The complete dataset with features and target species names
3. `model_metadata.json`: Feature names, target classes, dataset averages, and accuracy metrics
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def train_and_export_model():
    print("=" * 60)
    print("[*] Starting Iris Model Training...")
    print("=" * 60)

    # 1. Load Iris Dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = list(iris.feature_names)
    target_names = list(iris.target_names)

    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    df["species"] = df["target"].map(lambda idx: target_names[idx])

    print(f"[OK] Dataset loaded: {len(df)} samples across 3 classes: {target_names}")

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_names], df["target"], test_size=0.25, random_state=42, stratify=y
    )
    print(f"[OK] Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 3. Model Training
    model = RandomForestClassifier(
        n_estimators=100, max_depth=4, random_state=42
    )
    model.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    print(f"[OK] Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 5. Calculate Dataset Statistics (Averages & Class Averages)
    dataset_averages = df[feature_names].mean().to_dict()
    species_averages = (
        df.groupby("species")[feature_names]
        .mean()
        .to_dict(orient="index")
    )
    feature_ranges = {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
        }
        for col in feature_names
    }

    # 6. Save Artifacts
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "iris_model.pkl")
    data_csv_path = os.path.join(current_dir, "iris_data.csv")
    metadata_path = os.path.join(current_dir, "model_metadata.json")

    # Save model
    joblib.dump(model, model_path)
    print(f"[OK] Saved trained model: {model_path}")

    # Save dataset CSV
    df.to_csv(data_csv_path, index=False)
    print(f"[OK] Saved dataset CSV: {data_csv_path}")

    # Save metadata JSON
    metadata = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "accuracy": accuracy,
        "feature_names": feature_names,
        "target_names": target_names,
        "dataset_averages": dataset_averages,
        "species_averages": species_averages,
        "feature_ranges": feature_ranges,
        "confusion_matrix": conf_matrix,
        "classification_report": report,
    }

    def np_encoder(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, default=np_encoder)
    print(f"[OK] Saved model metadata: {metadata_path}")

    print("=" * 60)
    print("[SUCCESS] Training completed successfully!")
    print("=" * 60)

    return model, metadata, df


def load_or_train_model(force_retrain=False):
    """
    Loads the trained model, metadata, and dataset.
    If the files do not exist or force_retrain is True, runs train_and_export_model().
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "iris_model.pkl")
    metadata_path = os.path.join(current_dir, "model_metadata.json")
    data_csv_path = os.path.join(current_dir, "iris_data.csv")

    if force_retrain or not (os.path.exists(model_path) and os.path.exists(metadata_path) and os.path.exists(data_csv_path)):
        return train_and_export_model()

    model = joblib.load(model_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    df = pd.read_csv(data_csv_path)

    return model, metadata, df


if __name__ == "__main__":
    train_and_export_model()
