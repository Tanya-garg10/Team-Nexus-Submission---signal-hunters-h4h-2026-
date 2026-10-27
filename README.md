# 🔭 Signal Hunters: ML Challenge — Hack4Hack 2026, Round 2

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-HistGBT-orange?logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-scientific-013243?logo=numpy)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/submission-final-brightgreen)

> Classifies 6-panel ON/OFF radio cadences into **9 signal classes** using a drift-aware feature pipeline and a gradient-boosting ensemble, designed to handle the **GBT → Parkes (PKS) telescope domain shift**.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Signal Classes](#-signal-classes)
- [Approach](#-approach)
- [File Structure](#-file-structure)
- [How to Reproduce](#-how-to-reproduce)
- [Dependencies](#-dependencies)
- [Team](#-team)

---

## 📡 Problem Statement

Given raw radio telescope spectrograms recorded in a 6-panel ON/OFF cadence, classify each observation into one of 9 categories. The key challenge is **telescope domain shift**: training data comes from the **Green Bank Telescope (GBT)** while test data comes from the **Parkes Telescope (PKS)**.

---

## 🏷️ Signal Classes

| Class | Description |
|-------|-------------|
| `eti_candidate` | Extraterrestrial intelligence candidate signal |
| `frb` | Fast Radio Burst |
| `pulsar` | Rotating neutron star emission |
| `scintillating_star` | Interstellar scintillation from a star |
| `satellite_pass` | Satellite Radio Frequency Interference |
| `radar_chirp` | Chirped radar sweep |
| `rfi_intermittent` | Intermittent human-made RFI |
| `rfi_persistent` | Persistent human-made RFI |
| `noise` | Background noise (no signal) |

---

## 🧠 Approach

1. **Preprocessing** — Remove row glitches, smooth bandpass, normalise per panel, mask GBT centre frequency spike.
2. **Drift Search** — Per-panel matched filter over a grid of drift rates (`feats.py`).
3. **Cadence Logic** — Fine-grained linear track search over global time using `slew_gap_rows`; verifies ON panels lie on one track and OFF panels are empty (`feats2.py`).
4. **Broadband Features** — Simple dedispersion, channel coherence, patchiness metrics.
5. **Ensemble Model** — `HistGradientBoostingClassifier` (4 seeds), class-balanced.
6. **Domain Shift Mitigation** — Drop the most telescope-dependent features based on cross-telescope AUC probe.
7. **Self-Training (v2)** — Iteratively incorporate confident test predictions as pseudo-labels (`final2.py`).

---

## 🗂️ File Structure

```
📦 team-nexus/
 ┣ 📄 feats.py          # Per-panel drift-search feature extraction
 ┣ 📄 feats2.py         # Cadence track & broadband feature extraction
 ┣ 📄 run_feats.py      # Runs feats.py → saves F_train.npy, F_test.npy
 ┣ 📄 run_feats2.py     # Runs feats2.py → saves F2_train.npy, F2_test.npy
 ┣ 📄 final.py          # Trains ensemble, applies prior shift → submission v1
 ┣ 📄 submission.csv    # Final predictions for the test set
 ┗ 📄 README.md         # This file
```

---

## ▶️ How to Reproduce

**1. Setup dataset**

Unzip the competition dataset into the project root:
```
signal-hunters-round2/
```

**2. Extract features**
```bash
python run_feats.py      # per-panel drift features  (~2 min)
python run_feats2.py     # cadence track features    (~10 min on 1 CPU)
```

**3. Train & predict**
```bash
python final.py          # submission v1
```

> Only the provided competition data is used. No outside labels, no hand-labelling.

---

## 📦 Dependencies

```bash
pip install numpy scipy pandas scikit-learn
```

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations, drift grid |
| `scipy` | Median filter, signal processing |
| `pandas` | CSV I/O, metadata handling |
| `scikit-learn` | HistGradientBoostingClassifier, metrics |

---

## 👥 Team

**Team Nexus** — Hack4Hack 2026

| Name | Role |
|------|------|
| Tanya Garg | ML Pipeline & Feature Engineering |
| Arpan | Model Training & Submission |

---

<p align="center"><sub>Submission for Signal Hunters — Hack4Hack 2026 · Team Nexus</sub></p>
