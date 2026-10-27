# Team Nexus — Signal Hunters (H4H 2026)

**Hackathon:** Hack4Health 2026  
**Team:** Team Nexus  
**Competition:** Signal Hunters — Techno-India NJR Institute of Technology

---

## 📌 Problem Statement

Detect anomalous radio signals (potential extraterrestrial / non-RFI signals) from raw spectrogram data captured by telescopes. Each sample contains ON/OFF beam observations across multiple frequency channels and time steps.

---

## 🗂️ Project Structure

| File | Description |
|------|-------------|
| `feats.py` | Feature extraction pipeline — drift-corrected SNR, dedispersion, and spectral statistics for primary features |
| `feats2.py` | Secondary feature extraction — additional signal characteristics and cross-beam statistics |
| `run_feats.py` | Script to run `feats.py` on train/test data and save `F_train.npy`, `F_test.npy` |
| `run_feats2.py` | Script to run `feats2.py` on train/test data and save `F2_train.npy`, `F2_test.npy` |
| `final.py` | Model training — ensemble of `HistGradientBoostingClassifier` with EM-based label-prior shift correction |
| `submission.csv` | Final predictions for the test set |

---

## ⚙️ How to Run

### 1. Extract Features
```bash
python run_feats.py
python run_feats2.py
```

### 2. Train & Generate Predictions
```bash
python final.py
```

This will output `submission.csv` with predicted signal probabilities for each test sample.

---

## 🧠 Approach

- **Preprocessing:** Per-row median subtraction to remove broadband offsets; optional centre-spike masking for GBT data.
- **Feature Engineering:** Drift-corrected matched filters across a grid of drift rates (`feats.py`, `feats2.py`), ON/OFF beam ratio statistics, peak SNR, and spectral kurtosis.
- **Model:** Ensemble of 4 × `HistGradientBoostingClassifier` (seeds 0–3) with balanced class weights.
- **Post-processing:** EM-based label-prior shift (Saerens et al.) to correct for class imbalance between training distribution and test set.
- **Feature Selection:** Features with AUC > 0.62 on a held-out probe set are dropped to reduce leakage.

---

## 📦 Dependencies

```
numpy
scipy
pandas
scikit-learn
```

Install with:
```bash
pip install numpy scipy pandas scikit-learn
```

---

## 👥 Team Members

- Arpan
- Tanya Garg

---

> Submission for **Signal Hunters — H4H 2026**
