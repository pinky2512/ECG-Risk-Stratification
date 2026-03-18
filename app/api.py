"""
FastAPI backend for ECG Risk Stratification
Endpoints:
  POST /predict  — accepts .dat/.hea WFDB files, returns risk class + confidence
  GET  /health   — health check
"""

import os
import sys
import shutil
import tempfile
import numpy as np
import torch
import torch.nn.functional as F

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from models.cnn_transformer import CNNTransformer
from dataset import LABEL_MAP

import wfdb
import neurokit2 as nk

# ── App setup ──────────────────────────────────────────────
app = FastAPI(
    title="ECG Risk Stratification API",
    description="Predicts cardiovascular risk (Low/Medium/High) from 12-lead ECG signals",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model ─────────────────────────────────────────────
CKPT_PATH  = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "cnn_transformer_best.pt")
CLASSES    = ["Low", "Medium", "High"]
COLORS     = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
SAMPLING_RATE = 100

device = torch.device("cpu")
model  = CNNTransformer(num_classes=3)

if os.path.exists(CKPT_PATH):
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()
    print("✓ Model loaded successfully")
else:
    print(f"⚠ Checkpoint not found at {CKPT_PATH}")


# ── Response schema ────────────────────────────────────────
class PredictionResponse(BaseModel):
    risk_class:   str
    confidence:   float
    probabilities: dict
    color:        str
    message:      str


# ── Signal preprocessing ───────────────────────────────────
def preprocess_signal(signal: np.ndarray) -> torch.Tensor:
    """Bandpass filter + normalize → tensor (12, 1000)."""
    signal = signal.astype(np.float32)

    # Truncate or pad to 1000 timesteps
    if signal.shape[0] > 1000:
        signal = signal[:1000, :]
    elif signal.shape[0] < 1000:
        pad = np.zeros((1000 - signal.shape[0], 12), dtype=np.float32)
        signal = np.vstack([signal, pad])

    # Bandpass filter
    filtered = np.zeros_like(signal)
    for i in range(12):
        try:
            filtered[:, i] = nk.signal_filter(
                signal[:, i], sampling_rate=SAMPLING_RATE,
                lowcut=0.5, highcut=40.0, method="butterworth", order=4
            )
        except Exception:
            filtered[:, i] = signal[:, i]

    # Normalize per lead
    mean = filtered.mean(axis=0, keepdims=True)
    std  = filtered.std(axis=0, keepdims=True) + 1e-8
    normalized = (filtered - mean) / std

    return torch.tensor(normalized.T, dtype=torch.float32)  # (12, 1000)


# ── Endpoints ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "CNN-Transformer", "classes": CLASSES}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    dat_file: UploadFile = File(..., description=".dat WFDB signal file"),
    hea_file: UploadFile = File(..., description=".hea WFDB header file"),
):
    """
    Upload a WFDB ECG record (.dat + .hea files) and get risk prediction.
    """
    # Save uploaded files to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Strip extensions and save both files with same base name
        base = "record"
        dat_path = os.path.join(tmpdir, base + ".dat")
        hea_path = os.path.join(tmpdir, base + ".hea")

        dat_bytes = await dat_file.read()
        hea_bytes = await hea_file.read()

        with open(dat_path, "wb") as f:
            f.write(dat_bytes)

        # Fix header: replace original record name with "record"
        hea_text = hea_bytes.decode("utf-8", errors="ignore")
        lines    = hea_text.strip().split("\n")
        parts    = lines[0].split()
        parts[0] = base  # replace record name
        lines[0] = " ".join(parts)
        # Fix filename references in signal lines
        orig_base = os.path.splitext(dat_file.filename)[0]
        fixed_lines = []
        for line in lines:
            fixed_lines.append(line.replace(orig_base, base))
        with open(hea_path, "w") as f:
            f.write("\n".join(fixed_lines))

        try:
            record = wfdb.rdrecord(os.path.join(tmpdir, base))
            signal = record.p_signal
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read ECG file: {str(e)}")

        if signal.shape[1] != 12:
            raise HTTPException(status_code=400,
                                detail=f"Expected 12 leads, got {signal.shape[1]}")

        # Preprocess and predict
        x      = preprocess_signal(signal).unsqueeze(0)   # (1, 12, 1000)
        with torch.no_grad():
            logits = model(x)
            probs  = F.softmax(logits, dim=1)[0]

        pred_idx    = probs.argmax().item()
        pred_class  = CLASSES[pred_idx]
        confidence  = probs[pred_idx].item()
        prob_dict   = {c: round(probs[i].item(), 4) for i, c in enumerate(CLASSES)}

        messages = {
            "Low":    "No significant abnormality detected. Routine follow-up recommended.",
            "Medium": "Mild abnormality detected. Clinical monitoring advised.",
            "High":   "Significant abnormality detected. Urgent clinical evaluation recommended."
        }

        return PredictionResponse(
            risk_class=pred_class,
            confidence=round(confidence, 4),
            probabilities=prob_dict,
            color=COLORS[pred_class],
            message=messages[pred_class]
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)