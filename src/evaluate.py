"""
evaluate.py — Model Evaluation & Metrics Utilities
====================================================
Provides functions for evaluating trained models with multiple metrics,
generating confusion matrices, comparison tables, and visualizations.

Usage:
    from src.evaluate import evaluate_model, compare_models, plot_confusion_matrix
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    average: str = "weighted",
) -> dict:
    """
    Evaluate a trained model on test data with multiple metrics.

    Parameters
    ----------
    model : sklearn estimator
        Trained model with a .predict() method.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        True test labels.
    model_name : str
        Name of the model (for display).
    average : str, default "weighted"
        Averaging method for multi-class metrics.

    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average=average, zero_division=0),
    }

    # ROC-AUC (handle binary vs multi-class)
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            n_classes = len(np.unique(y_test))
            if n_classes == 2:
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(
                    y_test, y_proba, multi_class="ovr", average="macro"
                )
        else:
            metrics["roc_auc"] = None
    except Exception:
        metrics["roc_auc"] = None

    return metrics


def compare_models(results: dict, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """
    Evaluate all trained models and return a comparison DataFrame.

    Parameters
    ----------
    results : dict
        Output from train_all_models(): {model_name: {"model": ..., "training_time": ...}}
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        True test labels.

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by F1-score.
    """
    rows = []
    for name, result in results.items():
        metrics = evaluate_model(result["model"], X_test, y_test, model_name=name)
        metrics["training_time"] = result["training_time"]
        rows.append(metrics)

    df = pd.DataFrame(rows)
    df = df.sort_values("f1_score", ascending=False).reset_index(drop=True)

    return df


def print_classification_report(
    model, X_test: np.ndarray, y_test: np.ndarray, target_names: list = None
) -> str:
    """Print and return the sklearn classification report."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    print(report)
    return report


def plot_confusion_matrix(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: list = None,
    title: str = "Confusion Matrix",
    figsize: tuple = (8, 6),
    save_path: str = None,
) -> None:
    """
    Plot a confusion matrix heatmap.

    Parameters
    ----------
    model : sklearn estimator
        Trained model.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        True test labels.
    labels : list, optional
        Display labels for the axes.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save the figure.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Saved to {save_path}")

    plt.show()


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "f1_score",
    title: str = "Model Comparison",
    figsize: tuple = (10, 6),
    save_path: str = None,
) -> None:
    """
    Plot a bar chart comparing models on a given metric.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output from compare_models().
    metric : str
        Which metric column to plot.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette("viridis", n_colors=len(comparison_df))
    bars = ax.barh(comparison_df["model_name"], comparison_df[metric], color=colors)

    # Add value labels on bars
    for bar, val in zip(bars, comparison_df[metric]):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=10,
        )

    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.set_xlim(0, 1.1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Saved to {save_path}")

    plt.show()
