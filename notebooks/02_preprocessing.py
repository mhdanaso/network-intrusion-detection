"""
02_preprocessing.py — Data Cleaning & Feature Engineering
==========================================================
This notebook demonstrates the full preprocessing pipeline for the
NSL-KDD dataset, covering:
  1. Loading raw data
  2. Data quality checks
  3. Feature type analysis
  4. Label engineering (binary + multi-class)
  5. Encoding categorical features (One-Hot)
  6. Scaling numeric features (StandardScaler)
  7. Saving processed data for downstream modeling

Run:
    python notebooks/02_preprocessing.py
"""

import sys
from pathlib import Path

# Project root setup
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.data_loader import (
    load_train_data,
    load_test_data,
    add_attack_category,
    add_binary_label,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from src.preprocessing import build_preprocessing_pipeline, preprocess_data

FIGURES_DIR = ROOT / "reports" / "figures"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # ==============================================================
    # STEP 1: Load raw data
    # ==============================================================
    print("=" * 60)
    print("STEP 1: Loading raw NSL-KDD data")
    print("=" * 60)

    train_df = load_train_data()
    test_df = load_test_data()

    print(f"  Training set : {train_df.shape[0]:,} rows × {train_df.shape[1]} cols")
    print(f"  Test set     : {test_df.shape[0]:,} rows × {test_df.shape[1]} cols")

    # ==============================================================
    # STEP 2: Data quality checks
    # ==============================================================
    print(f"\n{'=' * 60}")
    print("STEP 2: Data quality checks")
    print("=" * 60)

    print(f"\n  Missing values (train): {train_df.isnull().sum().sum()}")
    print(f"  Missing values (test) : {test_df.isnull().sum().sum()}")
    print(f"  Duplicate rows (train): {train_df.duplicated().sum()}")
    print(f"  Duplicate rows (test) : {test_df.duplicated().sum()}")

    print("\n  Data types:")
    for dtype, count in train_df.dtypes.value_counts().items():
        print(f"    {dtype}: {count} columns")

    # ==============================================================
    # STEP 3: Feature type analysis
    # ==============================================================
    print(f"\n{'=' * 60}")
    print("STEP 3: Feature type analysis")
    print("=" * 60)

    print(f"\n  Categorical features ({len(CATEGORICAL_FEATURES)}):")
    for feat in CATEGORICAL_FEATURES:
        n_unique = train_df[feat].nunique()
        print(f"    {feat}: {n_unique} unique values")
        print(f"      Top 5: {train_df[feat].value_counts().head(5).to_dict()}")

    print(f"\n  Numeric features ({len(NUMERIC_FEATURES)}):")
    print(f"    {', '.join(NUMERIC_FEATURES[:10])}...")

    # Show summary statistics for key numeric features
    key_numeric = ["duration", "src_bytes", "dst_bytes", "count", "srv_count"]
    print(f"\n  Summary of key numeric features:")
    print(train_df[key_numeric].describe().round(2).to_string())

    # ==============================================================
    # STEP 4: Label engineering
    # ==============================================================
    print(f"\n{'=' * 60}")
    print("STEP 4: Label engineering")
    print("=" * 60)

    # Add labels to both sets
    train_df = add_attack_category(train_df)
    train_df = add_binary_label(train_df)
    test_df = add_attack_category(test_df)
    test_df = add_binary_label(test_df)

    print("\n  Binary label distribution (train):")
    binary_dist = train_df["binary_label"].value_counts()
    for label, count in binary_dist.items():
        name = "Normal" if label == 0 else "Attack"
        pct = count / len(train_df) * 100
        print(f"    {name} ({label}): {count:>7,} ({pct:.1f}%)")

    print("\n  Attack category distribution (train):")
    cat_dist = train_df["attack_category"].value_counts()
    for cat, count in cat_dist.items():
        pct = count / len(train_df) * 100
        print(f"    {cat:<8}: {count:>7,} ({pct:.1f}%)")

    # Plot class imbalance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors_binary = ["#2ecc71", "#e74c3c"]
    binary_dist.plot(kind="bar", ax=axes[0], color=colors_binary)
    axes[0].set_title("Binary Label Distribution")
    axes[0].set_xticklabels(["Normal (0)", "Attack (1)"], rotation=0)
    axes[0].set_ylabel("Count")

    colors_cat = sns.color_palette("Set2", n_colors=len(cat_dist))
    cat_dist.plot(kind="bar", ax=axes[1], color=colors_cat)
    axes[1].set_title("Attack Category Distribution")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "preprocessing_label_distribution.png", dpi=200)
    plt.close()
    print(f"\n  Saved label distribution plot.")

    # ==============================================================
    # STEP 5: Encoding categorical features
    # ==============================================================
    print(f"\n{'=' * 60}")
    print("STEP 5: Encoding categorical features (One-Hot)")
    print("=" * 60)

    print(f"\n  Before encoding:")
    print(f"    protocol_type unique: {train_df['protocol_type'].nunique()} "
          f"({train_df['protocol_type'].unique().tolist()})")
    print(f"    service unique:       {train_df['service'].nunique()}")
    print(f"    flag unique:          {train_df['flag'].nunique()}")

    preprocessor = build_preprocessing_pipeline()
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train_sample = train_df[feature_cols]

    X_train_encoded = preprocessor.fit_transform(X_train_sample)
    feature_names = preprocessor.get_feature_names_out().tolist()

    print(f"\n  After one-hot encoding:")
    print(f"    Original feature count:  {len(feature_cols)}")
    print(f"    Encoded feature count:   {X_train_encoded.shape[1]}")
    print(f"    New categorical columns: {X_train_encoded.shape[1] - len(NUMERIC_FEATURES)}")

    # ==============================================================
    # STEP 6: Scaling numeric features
    # ==============================================================
    print(f"\n{'=' * 60}")
    print("STEP 6: Scaling numeric features (StandardScaler)")
    print("=" * 60)

    # Show before/after for a few features
    sample_feats = ["duration", "src_bytes", "dst_bytes"]
    sample_indices = [NUMERIC_FEATURES.index(f) for f in sample_feats]

    print(f"\n  Before scaling (first 5 rows):")
    print(f"    {'Feature':<12} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    for feat in sample_feats:
        vals = train_df[feat]
        print(f"    {feat:<12} {vals.min():>12.2f} {vals.max():>12.2f} "
              f"{vals.mean():>12.2f} {vals.std():>12.2f}")

    print(f"\n  After scaling:")
    print(f"    {'Feature':<12} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    for feat, idx in zip(sample_feats, sample_indices):
        vals = X_train_encoded[:, idx]
        print(f"    {feat:<12} {vals.min():>12.4f} {vals.max():>12.4f} "
              f"{vals.mean():>12.4f} {vals.std():>12.4f}")

    # Plot distribution before/after scaling
    fig, axes = plt.subplots(len(sample_feats), 2, figsize=(14, 4 * len(sample_feats)))
    for i, (feat, idx) in enumerate(zip(sample_feats, sample_indices)):
        # Before
        axes[i, 0].hist(train_df[feat].clip(upper=train_df[feat].quantile(0.99)),
                        bins=50, color="#3498db", alpha=0.7)
        axes[i, 0].set_title(f"{feat} — Before Scaling")
        axes[i, 0].set_ylabel("Count")

        # After
        axes[i, 1].hist(X_train_encoded[:, idx], bins=50, color="#e74c3c", alpha=0.7)
        axes[i, 1].set_title(f"{feat} — After Scaling")
        axes[i, 1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "preprocessing_scaling_comparison.png", dpi=200)
    plt.close()
    print(f"\n  Saved scaling comparison plot.")

    # ==============================================================
    # STEP 7: Full pipeline — Binary classification
    # ==============================================================
    print(f"\n{'=' * 60}")
    print("STEP 7: Full preprocessing pipeline")
    print("=" * 60)

    # Binary
    print("\n  Processing for BINARY classification...")
    binary_data = preprocess_data(
        train_df, test_df, target_col="binary_label", label_type="binary"
    )
    print(f"    X_train: {binary_data['X_train'].shape}")
    print(f"    X_test:  {binary_data['X_test'].shape}")
    print(f"    y_train: {binary_data['y_train'].shape} "
          f"(classes: {np.unique(binary_data['y_train']).tolist()})")
    print(f"    y_test:  {binary_data['y_test'].shape}")

    # Multi-class
    print("\n  Processing for MULTI-CLASS classification...")
    multi_data = preprocess_data(
        train_df, test_df, target_col="attack_category", label_type="multiclass"
    )
    le = multi_data["label_encoder"]
    print(f"    X_train: {multi_data['X_train'].shape}")
    print(f"    X_test:  {multi_data['X_test'].shape}")
    print(f"    y_train: {multi_data['y_train'].shape} "
          f"(classes: {np.unique(multi_data['y_train']).tolist()})")
    print(f"    Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ==============================================================
    # Save processed data
    # ==============================================================
    print(f"\n{'=' * 60}")
    print("Saving processed data")
    print("=" * 60)

    np.save(PROCESSED_DIR / "X_train_binary.npy", binary_data["X_train"])
    np.save(PROCESSED_DIR / "X_test_binary.npy", binary_data["X_test"])
    np.save(PROCESSED_DIR / "y_train_binary.npy", binary_data["y_train"])
    np.save(PROCESSED_DIR / "y_test_binary.npy", binary_data["y_test"])

    np.save(PROCESSED_DIR / "X_train_multiclass.npy", multi_data["X_train"])
    np.save(PROCESSED_DIR / "X_test_multiclass.npy", multi_data["X_test"])
    np.save(PROCESSED_DIR / "y_train_multiclass.npy", multi_data["y_train"])
    np.save(PROCESSED_DIR / "y_test_multiclass.npy", multi_data["y_test"])

    # Save feature names for reference
    with open(PROCESSED_DIR / "feature_names.txt", "w") as f:
        for name in binary_data["feature_names"]:
            f.write(name + "\n")

    print(f"  Saved processed arrays to: {PROCESSED_DIR}")
    print(f"  Binary:     X_train={binary_data['X_train'].shape}, X_test={binary_data['X_test'].shape}")
    print(f"  Multiclass: X_train={multi_data['X_train'].shape}, X_test={multi_data['X_test'].shape}")

    # ==============================================================
    # Summary
    # ==============================================================
    print(f"\n{'=' * 60}")
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Original features:    {len(feature_cols)}")
    print(f"    Numeric:            {len(NUMERIC_FEATURES)}")
    print(f"    Categorical:        {len(CATEGORICAL_FEATURES)}")
    print(f"  Encoded features:     {binary_data['X_train'].shape[1]}")
    print(f"  Training samples:     {binary_data['X_train'].shape[0]:,}")
    print(f"  Test samples:         {binary_data['X_test'].shape[0]:,}")
    print(f"  Binary classes:       2 (Normal / Attack)")
    print(f"  Multi-class classes:  {len(le.classes_)} ({', '.join(le.classes_)})")
    print(f"\n  Saved figures to:     {FIGURES_DIR}")
    print(f"  Saved arrays to:      {PROCESSED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
