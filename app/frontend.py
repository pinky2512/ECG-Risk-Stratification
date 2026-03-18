"""
Streamlit frontend for ECG Risk Stratification
- Upload .dat + .hea WFDB files
- Display risk prediction + confidence
- Show raw ECG signal across all 12 leads
"""

import os
import sys
import requests
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import tempfile
import wfdb

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="ECG Risk Stratification",
    page_icon="🫀",
    layout="wide"
)

API_URL    = "http://localhost:8000"
CLASSES    = ["Low", "Medium", "High"]
COLORS     = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

# ── Header ─────────────────────────────────────────────────
st.title("🫀 ECG Cardiovascular Risk Stratification")
st.markdown("""
> **Deep learning model** that classifies 12-lead ECG signals into
> **Low / Medium / High** cardiovascular risk using a CNN-Transformer hybrid.
""")
st.divider()

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Model:** CNN-Transformer Hybrid  
    **Dataset:** PTB-XL (21,837 ECGs)  
    **Test Macro F1:** 0.6424  
    **High-Risk Recall:** 0.7591  

    **Risk Classes:**
    - 🟢 **Low** — No significant abnormality
    - 🟡 **Medium** — Mild abnormality, monitor
    - 🔴 **High** — Urgent evaluation needed

    **Top Predictive Leads:**  
    V1 > V2 > II > V3 > V4
    """)

    st.divider()
    st.markdown("**API Status**")
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            st.success("API Online ✓")
        else:
            st.error("API Error")
    except Exception:
        st.error("API Offline — start api.py first")

# ── File upload ────────────────────────────────────────────
st.subheader("📁 Upload ECG Files")
st.markdown("Upload a WFDB record — you need both the `.dat` and `.hea` files.")

col1, col2 = st.columns(2)
with col1:
    dat_file = st.file_uploader("Upload .dat file", type=["dat"])
with col2:
    hea_file = st.file_uploader("Upload .hea file", type=["hea"])

# ── Predict ────────────────────────────────────────────────
if dat_file and hea_file:
    st.divider()

    if st.button("🔍 Analyze ECG", type="primary", use_container_width=True):
        with st.spinner("Analyzing ECG signal..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    files={
                        "dat_file": (dat_file.name, dat_file.getvalue(), "application/octet-stream"),
                        "hea_file": (hea_file.name, hea_file.getvalue(), "application/octet-stream"),
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    risk   = result["risk_class"]
                    conf   = result["confidence"]
                    probs  = result["probabilities"]
                    color  = result["color"]
                    msg    = result["message"]

                    # ── Result display ─────────────────────
                    st.subheader("📊 Prediction Results")
                    r1, r2, r3 = st.columns(3)

                    with r1:
                        st.metric("Risk Class", risk)
                    with r2:
                        st.metric("Confidence", f"{conf*100:.1f}%")
                    with r3:
                        st.metric("Accuracy on Test Set", "66.81%")

                    # Color-coded result box
                    bg = {"Low": "#f0fdf4", "Medium": "#fffbeb", "High": "#fef2f2"}
                    st.markdown(f"""
                    <div style="padding:16px; border-radius:10px;
                                background:{bg[risk]}; border-left:5px solid {color};">
                        <h3 style="color:{color}; margin:0">{risk} Risk</h3>
                        <p style="margin:8px 0 0">{msg}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.divider()

                    # ── Probability bars ───────────────────
                    st.subheader("📈 Class Probabilities")
                    fig, ax = plt.subplots(figsize=(8, 2.5))
                    bars = ax.barh(
                        list(probs.keys()),
                        list(probs.values()),
                        color=[COLORS[c] for c in probs.keys()],
                        edgecolor="white", height=0.5
                    )
                    for bar, val in zip(bars, probs.values()):
                        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                                f"{val*100:.1f}%", va="center", fontsize=11)
                    ax.set_xlim(0, 1.15)
                    ax.set_xlabel("Probability")
                    ax.spines[["top","right","left"]].set_visible(False)
                    ax.tick_params(left=False)
                    plt.tight_layout()
                    st.pyplot(fig)

                    st.divider()

                    # ── ECG Signal plot ────────────────────
                    st.subheader("🔬 Raw ECG Signal — All 12 Leads")
                    try:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            # Save .dat file
                            with open(os.path.join(tmpdir, "rec.dat"), "wb") as f:
                                f.write(dat_file.getvalue())

                            # Fix header — replace original record name with "rec"
                            hea_content = hea_file.getvalue().decode("utf-8", errors="ignore")
                            orig_base   = os.path.splitext(dat_file.name)[0]
                            hea_fixed   = hea_content.replace(orig_base, "rec")
                            lines       = hea_fixed.split("\n")
                            parts       = lines[0].split()
                            parts[0]    = "rec"
                            lines[0]    = " ".join(parts)

                            with open(os.path.join(tmpdir, "rec.hea"), "w") as f:
                                f.write("\n".join(lines))

                            record = wfdb.rdrecord(os.path.join(tmpdir, "rec"))
                            signal = record.p_signal[:1000, :]

                        t = np.arange(signal.shape[0]) / 100
                        fig2, axes = plt.subplots(4, 3, figsize=(18, 10))
                        fig2.suptitle("12-Lead ECG", fontsize=14, fontweight="bold")

                        for i, (ax, lead) in enumerate(zip(axes.flat, LEAD_NAMES)):
                            ax.plot(t, signal[:, i], color="#334155", linewidth=0.8)
                            ax.set_title(lead, fontsize=10, fontweight="bold")
                            ax.set_xlabel("Time (s)", fontsize=8)
                            ax.spines[["top","right"]].set_visible(False)
                            ax.tick_params(labelsize=7)

                        plt.tight_layout()
                        st.pyplot(fig2)

                    except Exception as e:
                        st.warning(f"Could not render ECG plot: {str(e)}")

                else:
                    st.error(f"API Error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure api.py is running.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.info("👆 Upload both .dat and .hea files to get started.")

    # ── Sample files hint ──────────────────────────────────
    st.markdown("""
    **Where to find sample files:**  
    Your PTB-XL dataset at `data/ptbxl/records/records100/` contains thousands of sample records.  
    Each record has two files — e.g. `00001_lr.dat` and `00001_lr.hea`
    """)