"""
04_deep_learning.py — Neural Network Experiments
==================================================
This notebook implements a Multi-Layer Perceptron (MLP) using
TensorFlow/Keras for both binary and multi-class intrusion detection,
and compares its performance against the best traditional ML model.

Contents:
  1. Load preprocessed data
  2. Build MLP architecture
  3. Train with early stopping & learning rate scheduling
  4. Evaluate on test set
  5. Plot training history (loss & accuracy curves)
  6. Compare with best ML model from notebook 03
  7. Save the trained neural network

Run:
    python notebooks/04_deep_learning.py
"""

import sys
import os
import json
from pathlib import Path

# Project root setup
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Fix Windows terminal encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Suppress TensorFlow info/warning logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from src.data_loader import (
    load_train_data,
    load_test_data,
    add_attack_category,
    add_binary_label,
)
from src.preprocessing import preprocess_data

FIGURES_DIR = ROOT / "reports" / "figures"
MODELS_DIR = ROOT / "models"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_mlp(input_dim, num_classes, task="binary"):
    """
    Build a Multi-Layer Perceptron model.

    Architecture:
        Input → Dense(256, ReLU) → BatchNorm → Dropout(0.3)
              → Dense(128, ReLU) → BatchNorm → Dropout(0.3)
              → Dense(64, ReLU)  → BatchNorm → Dropout(0.2)
              → Output

    Parameters
    ----------
    input_dim : int
        Number of input features.
    num_classes : int
        Number of output classes.
    task : str
        "binary" or "multiclass" — determines output activation and loss.

    Returns
    -------
    keras.Model
        Compiled Keras model.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        # Hidden layer 1
        layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Hidden layer 2
        layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Hidden layer 3
        layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
    ])

    # Output layer
    if task == "binary":
        model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
    else:
        model.add(layers.Dense(num_classes, activation="softmax"))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    return model


def plot_training_history(history, task="binary"):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_title(f"Training & Validation Loss ({task})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[1].set_title(f"Training & Validation Accuracy ({task})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{task}_mlp_training_history.png", dpi=200)
    plt.close()


def evaluate_mlp(model, X_test, y_test, task="binary", label_encoder=None):
    """
    Evaluate the MLP model and return metrics dictionary.

    Returns
    -------
    dict
        Evaluation metrics matching the format from src.evaluate.
    """
    if task == "binary":
        y_proba = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_proba >= 0.5).astype(int)
        roc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_proba, axis=1)
        try:
            roc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        except ValueError:
            roc = None

    metrics = {
        "model_name": "MLP (Deep Learning)",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc,
    }

    # Print classification report
    if task == "binary":
        target_names = ["Normal", "Attack"]
    else:
        target_names = label_encoder.classes_.tolist() if label_encoder else None

    print(f"\n  Classification Report — MLP ({task}):")
    print(classification_report(y_test, y_pred,
                                target_names=target_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = target_names if target_names else sorted(np.unique(y_test).tolist())

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — MLP ({task.title()})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{task}_confusion_matrix_mlp.png", dpi=200)
    plt.close()

    return metrics, y_pred


def run_deep_learning_task(train_df, test_df, task="binary"):
    """
    Full deep learning pipeline for one task type.

    Parameters
    ----------
    task : str
        "binary" or "multiclass"

    Returns
    -------
    dict
        Results including metrics and training time.
    """
    task_label = "Binary (Normal vs Attack)" if task == "binary" else "Multi-class (5 categories)"
    target_col = "binary_label" if task == "binary" else "attack_category"
    label_type = "binary" if task == "binary" else "multiclass"

    print(f"\n{'#' * 60}")
    print(f"  DEEP LEARNING — {task_label}")
    print(f"{'#' * 60}")

    # ----------------------------------------------------------
    # 1. Preprocess
    # ----------------------------------------------------------
    print(f"\n--- Preprocessing ---")
    data = preprocess_data(train_df, test_df, target_col=target_col, label_type=label_type)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    label_encoder = data["label_encoder"]

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  Input dimension: {input_dim}")
    print(f"  Number of classes: {num_classes}")

    # ----------------------------------------------------------
    # 2. Build model
    # ----------------------------------------------------------
    print(f"\n--- Building MLP architecture ---")
    model = build_mlp(input_dim, num_classes, task=task)
    model.summary(print_fn=lambda x: print(f"  {x}"))

    # ----------------------------------------------------------
    # 3. Train with callbacks
    # ----------------------------------------------------------
    print(f"\n--- Training ---")

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=256,
        callbacks=cb_list,
        verbose=1,
    )
    training_time = time.time() - start_time

    print(f"\n  Training completed in {training_time:.2f}s")
    print(f"  Epochs run: {len(history.history['loss'])}")
    print(f"  Final train loss:     {history.history['loss'][-1]:.4f}")
    print(f"  Final val loss:       {history.history['val_loss'][-1]:.4f}")
    print(f"  Final train accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Final val accuracy:   {history.history['val_accuracy'][-1]:.4f}")

    # ----------------------------------------------------------
    # 4. Plot training history
    # ----------------------------------------------------------
    plot_training_history(history, task=task)
    print(f"  Saved training history plot.")

    # ----------------------------------------------------------
    # 5. Evaluate
    # ----------------------------------------------------------
    print(f"\n--- Evaluating on test set ---")
    metrics, y_pred = evaluate_mlp(model, X_test, y_test, task=task,
                                   label_encoder=label_encoder)
    metrics["training_time"] = training_time

    print(f"\n  MLP Results:")
    for key, val in metrics.items():
        if key == "model_name":
            continue
        if isinstance(val, float):
            print(f"    {key}: {val:.4f}")
        else:
            print(f"    {key}: {val}")

    # ----------------------------------------------------------
    # 6. Save model
    # ----------------------------------------------------------
    model_path = str(MODELS_DIR / f"{task}_mlp_model.keras")
    model.save(model_path)
    print(f"\n  Model saved to: {model_path}")

    return {
        "task": task,
        "metrics": metrics,
        "training_time": training_time,
        "history": history.history,
    }


def compare_with_ml_models(dl_results, task="binary"):
    """
    Load the saved ML results and compare with the MLP.

    Parameters
    ----------
    dl_results : dict
        Results from run_deep_learning_task().
    task : str
        "binary" or "multiclass"
    """
    results_path = MODELS_DIR / "training_results.json"
    if not results_path.exists():
        print("  (Skipping ML comparison — training_results.json not found)")
        return

    with open(results_path) as f:
        ml_results = json.load(f)

    if task not in ml_results:
        print(f"  (No {task} ML results to compare with)")
        return

    ml_comparison = ml_results[task]["comparison"]
    dl_metrics = dl_results["metrics"]

    # Build combined comparison table
    rows = []
    for entry in ml_comparison:
        rows.append({
            "Model": entry["model_name"],
            "Accuracy": entry["accuracy"],
            "Precision": entry["precision"],
            "Recall": entry["recall"],
            "F1 Score": entry["f1_score"],
            "ROC-AUC": entry.get("roc_auc", "N/A"),
            "Time (s)": entry["training_time"],
        })

    rows.append({
        "Model": "MLP (Deep Learning)",
        "Accuracy": dl_metrics["accuracy"],
        "Precision": dl_metrics["precision"],
        "Recall": dl_metrics["recall"],
        "F1 Score": dl_metrics["f1_score"],
        "ROC-AUC": dl_metrics.get("roc_auc", "N/A"),
        "Time (s)": dl_results["training_time"],
    })

    df = pd.DataFrame(rows)
    df = df.sort_values("F1 Score", ascending=False).reset_index(drop=True)

    task_label = "Binary" if task == "binary" else "Multi-class"
    print(f"\n{'─' * 90}")
    print(f"  ALL MODELS COMPARISON — {task_label}")
    print(f"{'─' * 90}")

    display = df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]:
        display[col] = display[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))
    display["Time (s)"] = display["Time (s)"].apply(lambda x: f"{x:.2f}s")
    print(display.to_string(index=False))
    print(f"{'─' * 90}")

    # Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("coolwarm", n_colors=len(df))
    bars = ax.barh(df["Model"], df["F1 Score"], color=colors)
    for bar, val in zip(bars, df["F1 Score"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    # Highlight MLP bar
    mlp_idx = df[df["Model"] == "MLP (Deep Learning)"].index
    if len(mlp_idx) > 0:
        bars[mlp_idx[0]].set_edgecolor("black")
        bars[mlp_idx[0]].set_linewidth(2)

    ax.set_xlabel("F1 Score")
    ax.set_title(f"ML vs Deep Learning — {task_label}")
    ax.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{task}_ml_vs_dl_comparison.png", dpi=200)
    plt.close()
    print(f"  Saved ML vs DL comparison chart.")


def main():
    print("=" * 60)
    print("DEEP LEARNING — Neural Network Experiments")
    print("=" * 60)

    # Check TensorFlow
    print(f"\n  TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"  GPU available: {gpus}")
    else:
        print(f"  GPU: Not available (using CPU)")

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------
    print(f"\n--- Loading NSL-KDD dataset ---")
    train_df = load_train_data()
    test_df = load_test_data()
    train_df = add_attack_category(train_df)
    train_df = add_binary_label(train_df)
    test_df = add_attack_category(test_df)
    test_df = add_binary_label(test_df)

    print(f"  Training: {train_df.shape[0]:,} samples")
    print(f"  Test:     {test_df.shape[0]:,} samples")

    # ----------------------------------------------------------
    # Binary classification
    # ----------------------------------------------------------
    binary_results = run_deep_learning_task(train_df, test_df, task="binary")
    compare_with_ml_models(binary_results, task="binary")

    # ----------------------------------------------------------
    # Multi-class classification
    # ----------------------------------------------------------
    multi_results = run_deep_learning_task(train_df, test_df, task="multiclass")
    compare_with_ml_models(multi_results, task="multiclass")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("DEEP LEARNING EXPERIMENTS COMPLETE")
    print("=" * 60)

    print(f"\n  Binary MLP:")
    bm = binary_results["metrics"]
    print(f"    Accuracy:  {bm['accuracy']:.4f}")
    print(f"    F1 Score:  {bm['f1_score']:.4f}")
    print(f"    ROC-AUC:   {bm['roc_auc']:.4f}" if bm['roc_auc'] else "    ROC-AUC:   N/A")
    print(f"    Time:      {binary_results['training_time']:.2f}s")

    print(f"\n  Multiclass MLP:")
    mm = multi_results["metrics"]
    print(f"    Accuracy:  {mm['accuracy']:.4f}")
    print(f"    F1 Score:  {mm['f1_score']:.4f}")
    print(f"    ROC-AUC:   {mm['roc_auc']:.4f}" if mm['roc_auc'] else "    ROC-AUC:   N/A")
    print(f"    Time:      {multi_results['training_time']:.2f}s")

    print(f"\n  KEY OBSERVATIONS:")
    print(f"  - MLP provides competitive performance with traditional ML models")
    print(f"  - Batch normalization and dropout help prevent overfitting")
    print(f"  - Early stopping ensures optimal generalization")
    print(f"  - XGBoost/Decision Tree remain strong for tabular data")
    print(f"  - Deep learning may shine more with larger, raw feature sets")

    print(f"\n  Figures saved to: {FIGURES_DIR}")
    print(f"  Models saved to:  {MODELS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
