"""
preprocessing.py — Data Preprocessing & Feature Engineering Pipeline
=====================================================================
Provides reusable preprocessing functions and scikit-learn Pipeline
for the NSL-KDD dataset.

Usage:
    from src.preprocessing import build_preprocessing_pipeline, preprocess_data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

from src.data_loader import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessing_pipeline(
    scale_numeric: bool = True,
    encode_categorical: bool = True,
    remove_low_variance: bool = True,
    variance_threshold: float = 0.01,
) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer for preprocessing the NSL-KDD dataset.

    Parameters
    ----------
    scale_numeric : bool, default True
        Whether to standardize numeric features.
    encode_categorical : bool, default True
        Whether to one-hot encode categorical features.
    remove_low_variance : bool, default True
        Whether to remove near-zero variance features.
    variance_threshold : float, default 0.01
        Threshold for VarianceThreshold (only used if remove_low_variance=True).

    Returns
    -------
    ColumnTransformer
        Fitted preprocessing transformer.
    """
    transformers = []

    # Numeric features — scale
    if scale_numeric:
        numeric_pipeline = Pipeline([
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, NUMERIC_FEATURES))
    else:
        transformers.append(("numeric", "passthrough", NUMERIC_FEATURES))

    # Categorical features — one-hot encode
    if encode_categorical:
        categorical_pipeline = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical_pipeline, CATEGORICAL_FEATURES))
    else:
        transformers.append(("categorical", "passthrough", CATEGORICAL_FEATURES))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # Drop columns not in either list (label, etc.)
    )

    return preprocessor


def encode_labels(y: pd.Series, label_type: str = "binary") -> tuple:
    """
    Encode target labels for classification.

    Parameters
    ----------
    y : pd.Series
        Target variable (attack_category or binary_label).
    label_type : str, default "binary"
        - "binary": Normal (0) vs Attack (1). Expects 'binary_label' column values.
        - "multiclass": Encodes attack categories as integers.

    Returns
    -------
    tuple
        (encoded_labels: np.ndarray, label_encoder: LabelEncoder or None)
    """
    if label_type == "binary":
        return y.values, None
    elif label_type == "multiclass":
        le = LabelEncoder()
        encoded = le.fit_transform(y)
        return encoded, le
    else:
        raise ValueError(f"Unknown label_type: {label_type}. Use 'binary' or 'multiclass'.")


def preprocess_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "binary_label",
    label_type: str = "binary",
):
    """
    Full preprocessing pipeline: fit on train, transform both train and test.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset (must include features + target column).
    test_df : pd.DataFrame
        Test dataset (must include features + target column).
    target_col : str, default "binary_label"
        Name of the target column.
    label_type : str, default "binary"
        Type of label encoding ("binary" or "multiclass").

    Returns
    -------
    dict
        {
            "X_train": np.ndarray,
            "X_test": np.ndarray,
            "y_train": np.ndarray,
            "y_test": np.ndarray,
            "preprocessor": ColumnTransformer,
            "label_encoder": LabelEncoder or None,
            "feature_names": list,
        }
    """
    # Separate features and target
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    # Build and fit preprocessor
    preprocessor = build_preprocessing_pipeline()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Encode labels
    y_train_encoded, label_encoder = encode_labels(y_train, label_type)
    if label_type == "multiclass" and label_encoder is not None:
        y_test_encoded = label_encoder.transform(y_test)
    else:
        y_test_encoded = y_test.values

    # Get feature names after transformation
    feature_names = preprocessor.get_feature_names_out().tolist()

    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train_encoded,
        "y_test": y_test_encoded,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
    }
