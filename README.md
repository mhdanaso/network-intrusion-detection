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
git clone https://github.com/<your-username>/network-intrusion-detection.git
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

### Run Notebooks
```bash
jupyter notebook
# Open notebooks in order: 01_eda → 02_preprocessing → 03_modeling → 04_deep_learning
```

### Run Dashboard
```bash
streamlit run app/app.py
```

---

## 📁 Project Structure

```
├── data/
│   ├── raw/                    # Original NSL-KDD dataset
│   └── processed/              # Cleaned & engineered features
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Data cleaning & feature engineering
│   ├── 03_modeling.ipynb       # Model training & evaluation
│   └── 04_deep_learning.ipynb  # Neural network experiments
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

> *Results will be added after model training is complete.*

<!-- 
| Model | Accuracy | F1-Score | ROC-AUC | Training Time |
|-------|----------|----------|---------|---------------|
| ... | ... | ... | ... | ... |
-->

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

<!-- Add your names and GitHub profiles -->
- **Person A** — Data pipeline, EDA, preprocessing, documentation
- **Person B** — Model training, evaluation, deep learning, dashboard

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html) — University of New Brunswick
- [KDD Cup 1999](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) — Original dataset
