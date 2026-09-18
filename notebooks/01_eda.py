"""
Exploratory Data Analysis (EDA) for NSL-KDD intrusion detection dataset.
This script performs the initial data quality and distribution analysis
for Person A on the feature/eda branch.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.data_loader import (
    load_train_data,
    add_attack_category,
    add_binary_label,
)


FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_distribution(series: pd.Series, title: str, filename: str, kind: str = "bar") -> None:
    """Save a distribution plot for a pandas Series."""
    plt.figure(figsize=(10, 5))
    if kind == "bar":
        sns.countplot(data=pd.DataFrame({title: series}), x=series, order=series.value_counts().index[:10])
        plt.xticks(rotation=45, ha="right")
    else:
        sns.histplot(series, bins=30, kde=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close()


def main() -> None:
    df = load_train_data(drop_difficulty=True)
    df = add_binary_label(df)
    df = add_attack_category(df)

    print("=== EDA SUMMARY ===")
    print(f"Shape: {df.shape}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Missing values: {int(df.isnull().sum().sum())}")
    print()

    print("Binary label distribution:")
    print(df["binary_label"].value_counts().to_string())
    print()

    print("Attack category distribution:")
    print(df["attack_category"].value_counts().to_string())
    print()

    print("Protocol type distribution:")
    print(df["protocol_type"].value_counts().to_string())
    print()

    print("Top 10 services:")
    print(df["service"].value_counts().head(10).to_string())
    print()

    print("Top 10 flags:")
    print(df["flag"].value_counts().head(10).to_string())
    print()

    numeric_cols = ["duration", "src_bytes", "dst_bytes", "count", "srv_count"]
    print("Numeric feature summary:")
    print(df[numeric_cols].describe().to_string())

    # Plot distributions
    plot_distribution(df["binary_label"], "Binary Label Distribution", "binary_label_distribution.png")
    plot_distribution(df["attack_category"], "Attack Category Distribution", "attack_category_distribution.png")
    plot_distribution(df["protocol_type"], "Protocol Type Distribution", "protocol_type_distribution.png")
    plot_distribution(df["service"], "Top Services", "service_distribution.png")

    # Numeric feature overview
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df[numeric_cols], orient="h", palette="Set2")
    plt.title("Numeric Feature Distribution Overview")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "numeric_feature_boxplot.png", dpi=200)
    plt.close()

    # Save a lightweight summary report
    summary_path = ROOT / "reports" / "eda_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("NSL-KDD EDA Summary\n")
        f.write("===================\n\n")
        f.write(f"Shape: {df.shape}\n")
        f.write(f"Duplicate rows: {df.duplicated().sum()}\n")
        f.write(f"Missing values: {int(df.isnull().sum().sum())}\n\n")
        f.write("Binary label distribution:\n")
        f.write(df["binary_label"].value_counts().to_string() + "\n\n")
        f.write("Attack category distribution:\n")
        f.write(df["attack_category"].value_counts().to_string() + "\n\n")
        f.write("Protocol type distribution:\n")
        f.write(df["protocol_type"].value_counts().to_string() + "\n\n")
        f.write("Dataset notes:\n")
        f.write("- The dataset has no missing values.\n")
        f.write("- Binary labels are roughly balanced, but attack subcategories are imbalanced.\n")
        f.write("- TCP dominates the protocol distribution, and HTTP/private are the most common services.\n")

    print(f"\nSaved plots to: {FIGURES_DIR}")
    print(f"Saved summary to: {summary_path}")
    print("\nEDA completed for Person A.")


if __name__ == "__main__":
    main()
