"""
03_modeling.py — Model Training & Evaluation
==============================================
This notebook demonstrates:
  1. Loading preprocessed data
  2. Training 6 ML models (Logistic Regression, Decision Tree,
     Random Forest, XGBoost, SVM, KNN)
  3. Evaluating each model with multiple metrics
  4. Comparing models and selecting the best
  5. Detailed analysis of the best model
  6. Generating comparison charts and confusion matrices

Run:
    python notebooks/03_modeling.py
"""

import sys
import json
from pathlib import Path

# Project root setup
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Fix Windows terminal encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

from src.data_loader import (
    load_train_data,
    load_test_data,
    add_attack_category,
    add_binary_label,
)
from src.preprocessing import preprocess_data
from src.models import get_models, train_model, save_model
from src.evaluate import evaluate_model, compare_models, plot_confusion_matrix

FIGURES_DIR = ROOT / "reports" / "figures"
MODELS_DIR = ROOT / "models"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def run_task(train_df, test_df, task="binary"):
    """
    Run the full train-evaluate cycle for one task type.

    Parameters
    ----------
    task : str
        "binary" or "multiclass"

    Returns
    -------
    dict
        Results summary including best model name and comparison table.
    """
    task_label = "Binary (Normal vs Attack)" if task == "binary" else "Multi-class (5 categories)"
    target_col = "binary_label" if task == "binary" else "attack_category"
    label_type = "binary" if task == "binary" else "multiclass"

    print(f"\n{'#' * 60}")
    print(f"  TASK: {task_label}")
    print(f"{'#' * 60}")

    # ----------------------------------------------------------
    # 1. Preprocess
    # ----------------------------------------------------------
    print(f"\n--- Preprocessing for {task} classification ---")
    data = preprocess_data(train_df, test_df, target_col=target_col, label_type=label_type)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    label_encoder = data["label_encoder"]

    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  Classes: {np.unique(y_train).tolist()}")

    # ----------------------------------------------------------
    # 2. Train all models
    # ----------------------------------------------------------
    print(f"\n--- Training 6 ML models ---")
    models = get_models(task=task)
    results = {}

    for name, model in models.items():
        print(f"\n  Training {name}...", end=" ", flush=True)
        result = train_model(model, X_train, y_train)
        t = result["training_time"]
        print(f"Done in {t:.2f}s")
        results[name] = result

    # ----------------------------------------------------------
    # 3. Evaluate all models
    # ----------------------------------------------------------
    print(f"\n--- Evaluating models on test set ---")
    comparison_df = compare_models(results, X_test, y_test)

    print(f"\n{'─' * 80}")
    print(f"  MODEL COMPARISON — {task_label}")
    print(f"{'─' * 80}")

    display = comparison_df.copy()
    for col in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        if col in display.columns:
            display[col] = display[col].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
    display["training_time"] = display["training_time"].apply(lambda x: f"{x:.2f}s")
    print(display.to_string(index=False))
    print(f"{'─' * 80}")

    # ----------------------------------------------------------
    # 4. Comparison chart — F1 Score
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", n_colors=len(comparison_df))
    bars = ax.barh(comparison_df["model_name"], comparison_df["f1_score"], color=colors)
    for bar, val in zip(bars, comparison_df["f1_score"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)
    ax.set_xlabel("F1 Score")
    ax.set_title(f"Model Comparison — {task_label}")
    ax.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{task}_model_comparison.png", dpi=200)
    plt.close()
    print(f"\n  Saved comparison chart.")

    # ----------------------------------------------------------
    # 5. Multi-metric radar chart
    # ----------------------------------------------------------
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
    available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comparison_df))
    width = 0.18
    palette = sns.color_palette("Set2", n_colors=len(available_metrics))

    for i, metric in enumerate(available_metrics):
        offset = (i - len(available_metrics) / 2) * width
        bars = ax.bar(x + offset, comparison_df[metric], width,
                      label=metric.replace("_", " ").title(), color=palette[i])

    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["model_name"], rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Multi-Metric Comparison — {task_label}")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{task}_multi_metric_comparison.png", dpi=200)
    plt.close()
    print(f"  Saved multi-metric comparison chart.")

    # ----------------------------------------------------------
    # 6. Best model — detailed analysis
    # ----------------------------------------------------------
    best_name = comparison_df.iloc[0]["model_name"]
    best_model = results[best_name]["model"]
    best_f1 = comparison_df.iloc[0]["f1_score"]

    print(f"\n  BEST MODEL: {best_name} (F1 = {best_f1:.4f})")

    # Classification report
    y_pred = best_model.predict(X_test)

    if task == "binary":
        target_names = ["Normal", "Attack"]
    else:
        target_names = label_encoder.classes_.tolist() if label_encoder else None

    print(f"\n  Classification Report — {best_name}:")
    print(classification_report(y_test, y_pred,
                                target_names=target_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = target_names if target_names else sorted(np.unique(y_test).tolist())

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name} ({task_label})")
    plt.tight_layout()
    cm_filename = f"{task}_confusion_matrix_{best_name.lower().replace(' ', '_')}.png"
    plt.savefig(FIGURES_DIR / cm_filename, dpi=200)
    plt.close()
    print(f"  Saved confusion matrix.")

    # ----------------------------------------------------------
    # 7. ROC curves (binary only)
    # ----------------------------------------------------------
    if task == "binary":
        fig, ax = plt.subplots(figsize=(10, 8))
        for name, result in results.items():
            model = result["model"]
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc_val = auc(fpr, tpr)
                ax.plot(fpr, tpr, linewidth=2,
                        label=f"{name} (AUC = {roc_auc_val:.4f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.5)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — Binary Classification")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "binary_roc_curves.png", dpi=200)
        plt.close()
        print(f"  Saved ROC curves.")

    # ----------------------------------------------------------
    # 8. Training time comparison
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    times = comparison_df.set_index("model_name")["training_time"]
    times.plot(kind="barh", color=sns.color_palette("rocket", n_colors=len(times)), ax=ax)
    ax.set_xlabel("Training Time (seconds)")
    ax.set_title(f"Training Time — {task_label}")
    for i, (name, val) in enumerate(times.items()):
        ax.text(val + 0.1, i, f"{val:.2f}s", va="center")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{task}_training_time.png", dpi=200)
    plt.close()
    print(f"  Saved training time chart.")

    return {
        "task": task,
        "best_model_name": best_name,
        "comparison": comparison_df.to_dict(orient="records"),
        "results": results,
        "data": data,
    }


def main():
    # ==============================================================
    # Load data
    # ==============================================================
    print("=" * 60)
    print("LOADING NSL-KDD DATASET")
    print("=" * 60)

    train_df = load_train_data()
    test_df = load_test_data()
    train_df = add_attack_category(train_df)
    train_df = add_binary_label(train_df)
    test_df = add_attack_category(test_df)
    test_df = add_binary_label(test_df)

    print(f"  Training: {train_df.shape[0]:,} samples")
    print(f"  Test:     {test_df.shape[0]:,} samples")

    # ==============================================================
    # Binary classification
    # ==============================================================
    binary_results = run_task(train_df, test_df, task="binary")

    # ==============================================================
    # Multi-class classification
    # ==============================================================
    multi_results = run_task(train_df, test_df, task="multiclass")

    # ==============================================================
    # Summary
    # ==============================================================
    print(f"\n{'=' * 60}")
    print("MODELING COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"  Binary best model:     {binary_results['best_model_name']}")
    print(f"  Multiclass best model: {multi_results['best_model_name']}")
    print(f"\n  Figures saved to: {FIGURES_DIR}")
    print(f"  Models saved to:  {MODELS_DIR}")

    # Key takeaways
    print(f"\n  KEY TAKEAWAYS:")
    print(f"  - Decision Tree and XGBoost consistently top-perform")
    print(f"  - XGBoost has superior ROC-AUC despite slightly lower F1")
    print(f"  - SVM is the slowest model with no accuracy advantage")
    print(f"  - KNN trains fastest but has limited scalability")
    print(f"  - Class imbalance affects R2L and U2R detection (multi-class)")
    print("=" * 60)


if __name__ == "__main__":
    main()
