# ECG-Risk-Stratification
ECG-based cardiovascular risk stratification using deep learning on PTB-XL


# ECG-Based Cardiovascular Risk Stratification Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

> Stratifying patients into **Low / Medium / High** cardiovascular risk from raw 12-lead ECG signals using deep learning on the PTB-XL dataset.


---

## 📌 Overview

Most ECG classifiers treat this as a binary or rhythm-detection problem. This project frames it as **clinical risk stratification** — directly useful for triage prioritization in resource-constrained settings.

**Prediction task:** Multiclass classification → Low / Medium / High cardiovascular risk  
**Dataset:** [PTB-XL](https://physionet.org/content/ptb-xl/) — 21,837 12-lead ECGs, 100/500 Hz  
**Architecture:** CNN-Transformer hybrid (1D CNN feature extractor + Transformer encoder)

---

## 🏷️ Label Mapping

| Risk Class | SCP-ECG Codes | Clinical Meaning |
|------------|---------------|-----------------|
| Low | NORM, SR | No significant abnormality |
| Medium | IRBBB, LAD, mild ST changes, early AFIB | Warrants monitoring |
| High | MI variants, LBBB, RBBB, ischemia, VT/VF | Requires urgent evaluation |

> Multi-label records: assigned the **highest** risk label (conservative clinical default).

---

## ⚙️ Setup

```bash
git clone https://github.com/YOUR_USERNAME/ecg-risk-stratification.git
cd ecg-risk-stratification
pip install -r requirements.txt
```

**Download PTB-XL:**
```bash
# Requires a free PhysioNet account
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/ -P data/ptbxl/
```

> Data is excluded from version control via `.gitignore`. ~2.5 GB download.


---

## 🔍 Key Design Decisions

- **Patient-level train/val/test split** — prevents data leakage from same-patient records appearing across splits
- **Worst-case label assignment** — multi-label records get the highest risk class (clinically conservative)
- **High-risk recall prioritized** — threshold tuned to minimize false negatives for high-risk class
- **Grad-CAM interpretability** — activations validated against known ECG anatomy (ST segments, QRS morphology)

---

## 📦 Requirements

```
torch>=2.0
wfdb
neurokit2
scikit-learn
numpy
pandas
matplotlib
seaborn
wandb
```

---

