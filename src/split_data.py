"""
Patient-level stratified train/val/test split.
Splits by patient_id to prevent data leakage — same patient
cannot appear in both train and test sets.
"""

import os
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────
DATA_PATH  = "data/ptbxl/records/"
SAVE_PATH  = "data/"

# ── Risk label mapping (same as EDA) ──────────────────────
LOW_RISK = {"NORM", "SR"}

MEDIUM_RISK = {
    "IRBBB", "LAD", "LAFB", "ISC_", "ISCA", "IVCD",
    "ABQRS", "PVC", "PAC", "SVTAC", "AFLT", "AFIB",
    "SARRH", "SBRAD", "STACH",
}

HIGH_RISK = {
    "AMI", "ASMI", "ILMI", "IPLMI", "IPMI", "ISCAL",
    "ISCAN", "ISCAS", "ISCIL", "ISCIN", "ISCLA",
    "LMI", "PMI", "LBBB", "RBBB", "LPFB",
    "AVB", "3AVB", "2AVB", "WPW", "VTACH", "VFIB",
    "STD_", "STE_", "NST_",
}

def assign_risk(scp_codes: dict) -> str:
    codes = set(scp_codes.keys())
    if codes & HIGH_RISK:   return "High"
    if codes & MEDIUM_RISK: return "Medium"
    return "Low"

# ── Load metadata ──────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA_PATH, "ptbxl_database.csv"), index_col="ecg_id")
df["scp_codes"]  = df["scp_codes"].apply(ast.literal_eval)
df["risk_label"] = df["scp_codes"].apply(assign_risk)

label_map = {"Low": 0, "Medium": 1, "High": 2}
df["label"] = df["risk_label"].map(label_map)

print(f"Total records   : {len(df):,}")
print(f"Unique patients : {df['patient_id'].nunique():,}")
print(f"\nClass distribution:\n{df['risk_label'].value_counts()}\n")

# ── Patient-level split ────────────────────────────────────
# Get one label per patient (worst-case across all their records)
patient_labels = (
    df.groupby("patient_id")["label"]
    .max()  # worst-case: if any record is High, patient = High
    .reset_index()
)

patients      = patient_labels["patient_id"].values
patient_strat = patient_labels["label"].values

# 70% train | 15% val | 15% test
train_pts, temp_pts, _, temp_strat = train_test_split(
    patients, patient_strat,
    test_size=0.30, random_state=42, stratify=patient_strat
)
val_pts, test_pts = train_test_split(
    temp_pts,
    test_size=0.50, random_state=42, stratify=temp_strat
)

# Map patients back to records
train_df = df[df["patient_id"].isin(train_pts)].copy()
val_df   = df[df["patient_id"].isin(val_pts)].copy()
test_df  = df[df["patient_id"].isin(test_pts)].copy()

# ── Verify no leakage ──────────────────────────────────────
assert len(set(train_pts) & set(val_pts))  == 0, "LEAKAGE: train/val overlap!"
assert len(set(train_pts) & set(test_pts)) == 0, "LEAKAGE: train/test overlap!"
assert len(set(val_pts)   & set(test_pts)) == 0, "LEAKAGE: val/test overlap!"
print("✓ No patient leakage detected across splits")

# ── Print split stats ──────────────────────────────────────
for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"\n{name} ({len(split):,} records, {split['patient_id'].nunique():,} patients):")
    dist = split["risk_label"].value_counts()
    for cls in ["Low", "Medium", "High"]:
        n = dist.get(cls, 0)
        print(f"  {cls:8s}: {n:5,}  ({n/len(split)*100:.1f}%)")

# ── Save splits ────────────────────────────────────────────
os.makedirs(SAVE_PATH, exist_ok=True)
train_df.to_csv(os.path.join(SAVE_PATH, "train.csv"))
val_df.to_csv(os.path.join(SAVE_PATH, "val.csv"))
test_df.to_csv(os.path.join(SAVE_PATH, "test.csv"))

print(f"\n✓ Splits saved to {SAVE_PATH}")
print("  train.csv / val.csv / test.csv")