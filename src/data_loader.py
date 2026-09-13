"""
data_loader.py — NSL-KDD Dataset Loading Utilities
====================================================
Provides functions to load the NSL-KDD dataset with proper column names,
basic validation, and attack category mapping.

Usage:
    from src.data_loader import load_train_data, load_test_data, load_data
"""

import os
import pandas as pd
import numpy as np


# ============================================================
# Column names for the NSL-KDD dataset (41 features + label + difficulty)
# ============================================================
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty_level"
]

# ============================================================
# Attack type → category mapping
# Maps each specific attack name to one of 5 categories
# ============================================================
ATTACK_CATEGORY_MAP = {
    # Normal
    "normal": "Normal",

    # DoS — Denial of Service
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "apache2": "DoS",
    "udpstorm": "DoS",
    "processtable": "DoS",
    "mailbomb": "DoS",

    # Probe — Surveillance/Scanning
    "satan": "Probe",
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "mscan": "Probe",
    "saint": "Probe",

    # R2L — Remote to Local (unauthorized remote access)
    "guess_passwd": "R2L",
    "ftp_write": "R2L",
    "imap": "R2L",
    "phf": "R2L",
    "multihop": "R2L",
    "warezmaster": "R2L",
    "warezclient": "R2L",
    "spy": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "snmpguess": "R2L",
    "snmpgetattack": "R2L",
    "httptunnel": "R2L",
    "sendmail": "R2L",
    "named": "R2L",
    "worm": "R2L",

    # U2R — User to Root (privilege escalation)
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "rootkit": "U2R",
    "perl": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
    "ps": "U2R",
    "httptunnel": "U2R",
}

# Categorical and numeric feature lists
CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]
NUMERIC_FEATURES = [col for col in COLUMN_NAMES if col not in CATEGORICAL_FEATURES + ["label", "difficulty_level"]]


# ============================================================
# Data loading functions
# ============================================================

def _get_data_path(filename: str) -> str:
    """Get the absolute path to a file in the data/raw/ directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "data", "raw", filename)


def load_data(filepath: str, drop_difficulty: bool = True) -> pd.DataFrame:
    """
    Load an NSL-KDD dataset file and assign proper column names.

    Parameters
    ----------
    filepath : str
        Path to the dataset file (.txt or .csv).
    drop_difficulty : bool, default True
        Whether to drop the 'difficulty_level' column (not useful for modeling).

    Returns
    -------
    pd.DataFrame
        Loaded dataset with proper column names.
    """
    df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)

    if drop_difficulty:
        df = df.drop(columns=["difficulty_level"], errors="ignore")

    # Strip whitespace from string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()

    return df


def load_train_data(drop_difficulty: bool = True) -> pd.DataFrame:
    """Load the full NSL-KDD training dataset (KDDTrain+.txt)."""
    filepath = _get_data_path("KDDTrain+.txt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Training data not found at {filepath}. "
            "Run 'python download_data.py' first."
        )
    return load_data(filepath, drop_difficulty)


def load_test_data(drop_difficulty: bool = True) -> pd.DataFrame:
    """Load the NSL-KDD test dataset (KDDTest+.txt)."""
    filepath = _get_data_path("KDDTest+.txt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Test data not found at {filepath}. "
            "Run 'python download_data.py' first."
        )
    return load_data(filepath, drop_difficulty)


def add_attack_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an 'attack_category' column mapping specific attacks to 5 categories:
    Normal, DoS, Probe, R2L, U2R.

    Unknown attack types are mapped to 'Unknown'.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with a 'label' column containing specific attack names.

    Returns
    -------
    pd.DataFrame
        Dataset with an added 'attack_category' column.
    """
    df = df.copy()
    df["attack_category"] = df["label"].map(ATTACK_CATEGORY_MAP).fillna("Unknown")
    return df


def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'binary_label' column: 0 for Normal, 1 for Attack.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with a 'label' column.

    Returns
    -------
    pd.DataFrame
        Dataset with an added 'binary_label' column.
    """
    df = df.copy()
    df["binary_label"] = (df["label"] != "normal").astype(int)
    return df


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Return a summary of the dataset including shape, dtypes, 
    missing values, and label distribution.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to summarize.

    Returns
    -------
    dict
        Summary statistics.
    """
    summary = {
        "shape": df.shape,
        "num_features": df.shape[1],
        "num_samples": df.shape[0],
        "missing_values": df.isnull().sum().sum(),
        "dtypes": df.dtypes.value_counts().to_dict(),
    }

    if "label" in df.columns:
        summary["label_distribution"] = df["label"].value_counts().to_dict()

    if "attack_category" in df.columns:
        summary["category_distribution"] = df["attack_category"].value_counts().to_dict()

    if "binary_label" in df.columns:
        summary["binary_distribution"] = df["binary_label"].value_counts().to_dict()

    return summary


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("Loading NSL-KDD training data...")
    try:
        train_df = load_train_data()
        train_df = add_attack_category(train_df)
        train_df = add_binary_label(train_df)

        print(f"\n✅ Training data loaded successfully!")
        print(f"   Shape: {train_df.shape}")
        print(f"   Columns: {list(train_df.columns)}")
        print(f"\n📊 Label Distribution:")
        print(train_df["label"].value_counts().head(10))
        print(f"\n📊 Attack Category Distribution:")
        print(train_df["attack_category"].value_counts())
        print(f"\n📊 Binary Label Distribution:")
        print(train_df["binary_label"].value_counts())

    except FileNotFoundError as e:
        print(f"❌ {e}")
