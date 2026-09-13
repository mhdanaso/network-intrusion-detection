"""
models.py — Model Definitions & Training Utilities
====================================================
Defines the ML models used for network intrusion detection
and provides training/saving utilities.

Usage:
    from src.models import get_models, train_model, save_model
"""

import os
import time
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


def get_models(task: str = "binary") -> dict:
    """
    Return a dictionary of ML models to train.

    Parameters
    ----------
    task : str, default "binary"
        - "binary": Models configured for binary classification.
        - "multiclass": Models configured for multi-class classification.

    Returns
    -------
    dict
        {model_name: model_instance}
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            max_depth=20,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="logloss" if task == "binary" else "mlogloss",
        ),
        "SVM": SVC(
            kernel="rbf",
            random_state=42,
            probability=True,  # Needed for ROC-AUC
            class_weight="balanced",
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1,
        ),
    }

    return models


def train_model(model, X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Train a single model and return results with timing.

    Parameters
    ----------
    model : sklearn estimator
        The model to train.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    dict
        {"model": fitted_model, "training_time": float (seconds)}
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    return {
        "model": model,
        "training_time": training_time,
    }


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str = "binary",
) -> dict:
    """
    Train all models and return results.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    task : str, default "binary"
        Classification task type.

    Returns
    -------
    dict
        {model_name: {"model": fitted_model, "training_time": float}}
    """
    models = get_models(task)
    results = {}

    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        result = train_model(model, X_train, y_train)
        print(f"   ✅ Done in {result['training_time']:.2f}s")
        results[name] = result

    return results


def save_model(model, filepath: str) -> None:
    """Save a trained model to disk using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"💾 Model saved to {filepath}")


def load_model(filepath: str):
    """Load a trained model from disk."""
    return joblib.load(filepath)
