# 🛡️ ML-Based Network Intrusion Detection System

A machine learning-powered system that detects malicious network traffic and classifies different types of cyber attacks using the **NSL-KDD** dataset.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-red?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Overview

Network Intrusion Detection Systems (NIDS) are critical for identifying malicious activity in network traffic. This project implements a **machine learning pipeline** that:

1. **Preprocesses** network traffic features (41 features from NSL-KDD)
2. **Trains** multiple ML models for binary and multi-class classification
3. **Evaluates** model performance with detailed metrics
4. **Deploys** the best model via an interactive Streamlit dashboard

### Attack Categories Detected

| Category | Description | Examples |
|----------|-------------|----------|
| **Normal** | Legitimate traffic | — |
| **DoS** | Denial of Service | neptune, smurf, back |
| **Probe** | Surveillance/Scanning | portsweep, ipsweep, nmap |
| **R2L** | Remote to Local | guess_passwd, ftp_write |
| **U2R** | User to Root | buffer_overflow, rootkit |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/mhdanaso/network-intrusion-detection.git
cd network-intrusion-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Download the NSL-KDD dataset
python download_data.py
```

### Run the experiment scripts
```bash
# Run these from the project root, after downloading the dataset:
python notebooks/01_eda.py
python notebooks/02_preprocessing.py
python notebooks/03_modeling.py
python notebooks/04_deep_learning.py
```

The scripts run the same stages as the original notebook workflow:
exploratory analysis, preprocessing, traditional ML modeling, and deep learning.
They save generated figures under `reports/figures/`. The full training scripts
can take several minutes, especially for SVM and deep learning.

### Run Dashboard
```bash
# Train models first (this creates the model artifacts used by the dashboard)
python -m src.train --task both

# Start the dashboard from the project root
streamlit run app/app.py
```

The dashboard displays the saved evaluation results and accepts headered CSV files
containing the 41 NSL-KDD feature columns. Predictions are processed in batches
(2,000 rows by default); reduce the batch size in the dashboard to lower inference
memory usage. The original input columns are included in the downloadable results,
along with predicted labels and, when available, prediction confidence.

The dashboard is for NSL-KDD-compatible benchmark data, not a live network sensor.
Streamlit keeps uploaded files in memory, so configure its upload limit appropriately
for your deployment and available memory.

### Input CSV format

The prediction page accepts a headered CSV containing the 41 NSL-KDD feature
columns. The required header and one valid example row are provided in
[`examples/sample_input.csv`](examples/sample_input.csv). The CSV may include
additional columns; they are preserved in the downloaded output. Do not include
`predicted_label` or `prediction_confidence`, because those names are reserved
for generated output columns.

### Run Tests
```bash
python -m unittest discover -s tests -v
```

To validate a fresh clone without downloading or retraining models, run the
test command above. The tests use small in-memory fixtures and do not require
the NSL-KDD dataset or saved model artifacts.

---

## 📁 Project Structure

```
├── data/
│   ├── raw/                    # Original NSL-KDD dataset
│   └── processed/              # Cleaned & engineered features
├── examples/
│   └── sample_input.csv        # Valid dashboard input example
├── notebooks/
│   ├── 01_eda.py               # Exploratory Data Analysis
│   ├── 02_preprocessing.py     # Data cleaning & feature engineering
│   ├── 03_modeling.py          # Model training & evaluation
│   └── 04_deep_learning.py     # Neural network experiments
├── src/
│   ├── data_loader.py          # Dataset loading utilities
│   ├── preprocessing.py        # Feature engineering pipeline
│   ├── models.py               # Model definitions & training
│   ├── evaluate.py             # Metrics & visualization
│   └── predict.py              # Inference module
├── app/
│   └── app.py                  # Streamlit web dashboard
├── models/                     # Saved trained models
├── reports/figures/             # Generated plots
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📊 Dataset

**NSL-KDD** — An improved version of the KDD Cup 1999 dataset for network intrusion detection benchmarking.

- **Training samples**: ~125,973
- **Test samples**: ~22,544
- **Features**: 41 (network connection attributes)
- **Source**: [UNB - NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)

---

## 🤖 Models

| Model | Task | Description |
|-------|------|-------------|
| Logistic Regression | Binary / Multi-class | Baseline, interpretable |
| Decision Tree | Binary / Multi-class | Explainable, fast |
| Random Forest | Binary / Multi-class | Ensemble, handles imbalance |
| XGBoost | Binary / Multi-class | State-of-the-art for tabular data |
| SVM | Binary / Multi-class | Classic ML approach |
| KNN | Binary / Multi-class | Instance-based learning |
| MLP (Deep Learning) | Binary / Multi-class | Neural network baseline |

---

## 📈 Results

### Binary Classification (Normal vs Attack)

| Model | Accuracy | F1-Score | ROC-AUC | Training Time |
|-------|----------|----------|---------|---------------|
| **Decision Tree** | 0.7915 | **0.7899** | 0.8133 | 6.48s |
| XGBoost | 0.7864 | 0.7844 | **0.9646** | 5.25s |
| KNN | 0.7803 | 0.7778 | 0.8274 | 0.05s |
| Random Forest | 0.7724 | 0.7693 | 0.9605 | 10.92s |
| Logistic Regression | 0.7540 | 0.7521 | 0.7921 | 12.59s |
| SVM | 0.7498 | 0.7476 | 0.7893 | 82.07s |

> 🏆 **Best Binary Model**: Decision Tree (F1 = 0.7899)

### Multi-class Classification (5 Attack Categories)

| Model | Accuracy | F1-Score | ROC-AUC | Training Time |
|-------|----------|----------|---------|---------------|
| **XGBoost** | 0.7787 | **0.7421** | **0.9506** | 18.34s |
| Decision Tree | 0.7685 | 0.7332 | 0.6494 | 5.23s |
| KNN | 0.7660 | 0.7212 | 0.7414 | 0.04s |
| Logistic Regression | 0.7613 | 0.7167 | 0.8794 | 39.09s |
| Random Forest | 0.7468 | 0.6986 | 0.8782 | 6.95s |
| SVM | 0.7357 | 0.6886 | 0.8479 | 549.79s |

> 🏆 **Best Multi-class Model**: XGBoost (F1 = 0.7421)

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **ML**: scikit-learn, XGBoost
- **Deep Learning**: TensorFlow / Keras
- **Data**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Dashboard**: Streamlit
- **Collaboration**: Git + GitHub

---

## 👥 Contributors

Contributions and improvements are welcome through GitHub pull requests.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html) — University of New Brunswick
- [KDD Cup 1999](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) — Original dataset
