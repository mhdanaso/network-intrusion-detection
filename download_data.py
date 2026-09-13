"""
download_data.py — NSL-KDD Dataset Download Script
===================================================
Downloads the NSL-KDD dataset files from the UNB repository
and saves them to the data/raw/ directory.

Usage:
    python download_data.py
"""

import os
import requests
from tqdm import tqdm


# NSL-KDD dataset URLs (hosted on UNB/alternative mirrors)
DATASET_URLS = {
    "KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
    "KDDTest+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
    "KDDTrain+_20Percent.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B_20Percent.txt",
    "KDDTest-21.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest-21.txt",
}

# NSL-KDD column names (41 features + label + difficulty_level)
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

# Output directory
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")


def download_file(url: str, filepath: str) -> None:
    """Download a file from a URL with a progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    filename = os.path.basename(filepath)

    with open(filepath, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def main():
    """Download all NSL-KDD dataset files."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("  NSL-KDD Dataset Downloader")
    print("=" * 60)
    print(f"\nTarget directory: {RAW_DATA_DIR}\n")

    for filename, url in DATASET_URLS.items():
        filepath = os.path.join(RAW_DATA_DIR, filename)

        if os.path.exists(filepath):
            print(f"[SKIP] {filename} already exists.")
            continue

        print(f"\n[DOWNLOADING] {filename}")
        print(f"  URL: {url}")

        try:
            download_file(url, filepath)
            print(f"[OK] Saved to {filepath}")
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to download {filename}: {e}")
            print("  Try downloading manually from: https://www.unb.ca/cic/datasets/nsl.html")

    print("\n" + "=" * 60)
    print("  Download complete!")
    print("=" * 60)
    print("\nNext step: Run notebooks/01_eda.ipynb to explore the data.")


if __name__ == "__main__":
    main()
