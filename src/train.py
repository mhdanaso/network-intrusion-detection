"""
train.py — Standalone Training Script
=======================================
Run the full ML pipeline from the command line:
    python -m src.train --task both       # Train binary + multi-class (default)
    python -m src.train --task binary     # Train only binary models
    python -m src.train --task multiclass # Train only multi-class models

What this script does (in simple terms):
1. Loads the NSL-KDD network traffic dataset
2. Prepares the data (scaling numbers, encoding categories)
3. Trains 6 different ML models
4. Checks which model is the best
5. Saves the winner so we can use it later in the app
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import seaborn as sns

# Fix Windows terminal encoding (cp1252 can't print emoji/unicode)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path so we can import our src modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_train_data, load_test_data, add_attack_category, add_binary_label
from src.preprocessing import preprocess_data
from src.models import get_models, train_model, save_model
from src.evaluate import evaluate_model, compare_models


# ============================================================
# Directory setup
# ============================================================
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_and_prepare_data():
    """
    Load the NSL-KDD dataset and add the label columns we need.

    Returns train and test DataFrames with:
    - 'binary_label': 0 = Normal, 1 = Attack
    - 'attack_category': Normal / DoS / Probe / R2L / U2R
    """
    print("=" * 60)
    print("📂 STEP 1: Loading the NSL-KDD dataset")
    print("=" * 60)

    train_df = load_train_data()
    test_df = load_test_data()

    # Add label columns
    train_df = add_attack_category(train_df)
    train_df = add_binary_label(train_df)
    test_df = add_attack_category(test_df)
    test_df = add_binary_label(test_df)

    print(f"   ✅ Training data: {train_df.shape[0]:,} samples, {train_df.shape[1]} columns")
    print(f"   ✅ Test data:     {test_df.shape[0]:,} samples, {test_df.shape[1]} columns")
    print(f"\n   📊 Binary label distribution (train):")
    print(f"      Normal: {(train_df['binary_label'] == 0).sum():,}")
    print(f"      Attack: {(train_df['binary_label'] == 1).sum():,}")
    print(f"\n   📊 Attack category distribution (train):")
    for cat, count in train_df["attack_category"].value_counts().items():
        print(f"      {cat}: {count:,}")

    return train_df, test_df


def run_training_pipeline(train_df, test_df, task="binary"):
    """
    Run the full pipeline for one task type (binary or multiclass).

    Steps:
    1. Preprocess data (scale + encode)
    2. Train all 6 models
    3. Evaluate each model
    4. Save the best one

    Parameters
    ----------
    task : str
        "binary" — Normal vs Attack (2 classes)
        "multiclass" — Normal / DoS / Probe / R2L / U2R (5 classes)
    """
    task_display = "Binary (Normal vs Attack)" if task == "binary" else "Multi-class (5 categories)"

    print(f"\n{'=' * 60}")
    print(f"🔧 TRAINING: {task_display}")
    print(f"{'=' * 60}")

    # --- Step 1: Preprocess ---
    print(f"\n📦 Preprocessing data for {task} classification...")

    if task == "binary":
        target_col = "binary_label"
        label_type = "binary"
    else:
        target_col = "attack_category"
        label_type = "multiclass"

    data = preprocess_data(train_df, test_df, target_col=target_col, label_type=label_type)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    preprocessor = data["preprocessor"]
    label_encoder = data["label_encoder"]

    print(f"   ✅ X_train shape: {X_train.shape}")
    print(f"   ✅ X_test shape:  {X_test.shape}")
    print(f"   ✅ Features after encoding: {X_train.shape[1]}")

    # --- Step 2: Train all models ---
    print(f"\n🏋️ Training 6 ML models...")
    models = get_models(task=task)
    results = {}

    for name, model in models.items():
        print(f"\n   🔧 Training {name}...", end=" ", flush=True)
        result = train_model(model, X_train, y_train)
        print(f"✅ Done in {result['training_time']:.2f}s")
        results[name] = result

    # --- Step 3: Evaluate all models ---
    print(f"\n📊 Evaluating models on test set...")
    comparison_df = compare_models(results, X_test, y_test)

    print(f"\n{'─' * 80}")
    print(f"📊 MODEL COMPARISON — {task_display}")
    print(f"{'─' * 80}")
    # Format the table nicely
    display_df = comparison_df.copy()
    for col in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.4f}" if x is not None else "N/A"
            )
    display_df["training_time"] = display_df["training_time"].apply(lambda x: f"{x:.2f}s")
    print(display_df.to_string(index=False))
    print(f"{'─' * 80}")

    # --- Step 4: Save comparison chart ---
    fig_path = os.path.join(FIGURES_DIR, f"{task}_model_comparison.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", n_colors=len(comparison_df))
    bars = ax.barh(comparison_df["model_name"], comparison_df["f1_score"], color=colors)
    for bar, val in zip(bars, comparison_df["f1_score"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)
    ax.set_xlabel("F1 Score")
    ax.set_title(f"Model Comparison — {task_display}")
    ax.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n   📊 Comparison chart saved to: {fig_path}")

    # --- Step 5: Save confusion matrix for best model ---
    best_name = comparison_df.iloc[0]["model_name"]
    best_model = results[best_name]["model"]
    print(f"\n   🏆 Best model: {best_name} (F1 = {comparison_df.iloc[0]['f1_score']:.4f})")

    from sklearn.metrics import confusion_matrix as cm_func
    y_pred = best_model.predict(X_test)
    cm = cm_func(y_test, y_pred)

    if task == "binary":
        labels = ["Normal", "Attack"]
    else:
        labels = label_encoder.classes_.tolist() if label_encoder else sorted(np.unique(y_test).tolist())

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name} ({task_display})")
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, f"{task}_confusion_matrix_{best_name.lower().replace(' ', '_')}.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Confusion matrix saved to: {cm_path}")

    # --- Step 6: Save the best model + preprocessor ---
    model_path = os.path.join(MODELS_DIR, f"{task}_best_model.joblib")
    preprocessor_path = os.path.join(MODELS_DIR, f"{task}_preprocessor.joblib")
    save_model(best_model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    print(f"   💾 Preprocessor saved to: {preprocessor_path}")

    if label_encoder is not None:
        le_path = os.path.join(MODELS_DIR, f"{task}_label_encoder.joblib")
        joblib.dump(label_encoder, le_path)
        print(f"   💾 Label encoder saved to: {le_path}")

    return {
        "task": task,
        "best_model_name": best_name,
        "comparison": comparison_df.to_dict(orient="records"),
    }


def main():
    """Main entry point — parse args and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Train ML models for Network Intrusion Detection"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="both",
        choices=["binary", "multiclass", "both"],
        help="Which classification task to run (default: both)",
    )
    args = parser.parse_args()

    # Load data once (used for both tasks)
    train_df, test_df = load_and_prepare_data()

    all_results = {}

    if args.task in ("binary", "both"):
        result = run_training_pipeline(train_df, test_df, task="binary")
        all_results["binary"] = result

    if args.task in ("multiclass", "both"):
        result = run_training_pipeline(train_df, test_df, task="multiclass")
        all_results["multiclass"] = result

    # Save combined results as JSON
    results_path = os.path.join(MODELS_DIR, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n📄 Full results saved to: {results_path}")

    print(f"\n{'=' * 60}")
    print("🎉 ALL DONE! Summary:")
    print(f"{'=' * 60}")
    for task_name, result in all_results.items():
        print(f"   {task_name.upper()}: Best model = {result['best_model_name']}")
    print(f"\n   📁 Models saved in:  {MODELS_DIR}")
    print(f"   📊 Figures saved in: {FIGURES_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
