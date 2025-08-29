# Hybrid IDS (DFA + ANN) — Streamlit App

A **self-contained** Streamlit interface that demonstrates a **hybrid intrusion detection system**:
- **DFA** for signature-like pattern flags (Rapid SYN & RST Scan examples)
- **ANN (MLPClassifier)** for anomaly detection trained on **NSL-KDD** (auto-fallback to synthetic)

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

> First run will attempt to download NSL-KDD. If download fails, it **falls back to a synthetic dataset** so everything still works.

## 📂 Project Structure

```
.
├─ app.py                # Streamlit UI
├─ dfa.py                # DFA engines and checks
├─ data_utils.py         # Dataset download/synthesis and prep
├─ model_utils.py        # Training, saving, loading, predicting
├─ artifacts/            # Saved model + metrics
├─ sample_data/          # Example CSVs (optional)
└─ requirements.txt
```

## 🧠 Features

- Train/Rebuild model (NSL-KDD or Synthetic)
- DFA checker for event sequences
- Single flow prediction
- Batch CSV prediction + download results
- Metrics (AUC, confusion matrix, classification report as JSON)

## 📑 CSV Schema for Batch
`duration, protocol_type, service, flag, src_bytes, dst_bytes, count, srv_count, same_srv_rate, dst_host_count, dst_host_srv_count`

## 🛠 Notes
- Replace/extend DFA patterns in `dfa.py`.
- Swap to CICIDS2017 by editing `data_utils.py` to load your CSV and keeping the same columns as this demo.