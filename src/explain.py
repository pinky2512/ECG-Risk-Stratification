"""
Interpretability for ECG Risk Stratification
1. Grad-CAM  — highlights which time regions drove the prediction
2. Lead Importance — which of the 12 leads matter most
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import wfdb

sys.path.insert(0, os.path.dirname(__file__))
from dataset import PTBXLDataset, LABEL_MAP
from models.cnn_transformer import CNNTransformer

CLASSES    = ["Low", "Medium", "High"]
LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
COLORS     = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}

# ── Grad-CAM for 1D CNN ────────────────────────────────────
class GradCAM1D:
    """
    Computes Grad-CAM activation maps for the last CNN conv layer.
    Works on CNNTransformer's CNN extractor.
    """
    def __init__(self, model: torch.nn.Module):
        self.model      = model
        self.gradients  = None
        self.activations = None
        # Hook onto last conv layer in CNN extractor
        target = model.cnn.cnn[-3]   # last Conv1d (Block 3)
        target.register_forward_hook(self._save_activation)
        target.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def compute(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Returns CAM of shape (T,) — same time length as CNN output (~125).
        """
        self.model.eval()
        x = x.unsqueeze(0).requires_grad_(True)   # (1, 12, 1000)

        logits = self.model(x)                     # (1, 3)
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global average pool gradients over channels
        weights = self.gradients.mean(dim=-1)      # (1, C)
        cam     = (weights.unsqueeze(-1) * self.activations).sum(dim=1)  # (1, T)
        cam     = F.relu(cam).squeeze(0).numpy()   # (T,)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def upsample_cam(cam: np.ndarray, target_len: int = 1000) -> np.ndarray:
    """Upsample CAM from ~125 to 1000 (original signal length)."""
    x_old = np.linspace(0, 1, len(cam))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, cam)


# ── Grad-CAM Visualization ─────────────────────────────────
def plot_gradcam(model, dataset, indices, save_dir="results/figures/"):
    """
    For each sample index, plot all 12 leads with Grad-CAM overlay.
    """
    os.makedirs(save_dir, exist_ok=True)
    cam_engine = GradCAM1D(model)
    inv_map    = {v: k for k, v in LABEL_MAP.items()}

    for idx in indices:
        x, y      = dataset[idx]
        true_label = inv_map[y.item()]
        pred_logit = model(x.unsqueeze(0))
        pred_idx   = pred_logit.argmax(1).item()
        pred_label = CLASSES[pred_idx]
        confidence = F.softmax(pred_logit, dim=1)[0, pred_idx].item()

        cam     = cam_engine.compute(x, pred_idx)
        cam_up  = upsample_cam(cam, target_len=1000)
        signal  = x.numpy()   # (12, 1000)
        t       = np.arange(1000) / 100  # time in seconds

        fig = plt.figure(figsize=(20, 14))
        fig.suptitle(
            f"Grad-CAM | True: {true_label}  Predicted: {pred_label} "
            f"({confidence*100:.1f}%)",
            fontsize=14, fontweight="bold"
        )
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.3)

        for i, lead in enumerate(LEAD_NAMES):
            ax  = fig.add_subplot(gs[i // 3, i % 3])
            sig = signal[i]

            # Plot raw signal
            ax.plot(t, sig, color="#334155", linewidth=0.8, zorder=2)

            # Overlay Grad-CAM heatmap
            for j in range(len(t) - 1):
                ax.axvspan(t[j], t[j+1], alpha=cam_up[j] * 0.6,
                           color=COLORS[pred_label], zorder=1)

            ax.set_title(lead, fontsize=9, fontweight="bold")
            ax.set_xlabel("Time (s)", fontsize=7)
            ax.set_xticks([0, 2, 4, 6, 8, 10])
            ax.tick_params(labelsize=7)
            ax.spines[["top","right"]].set_visible(False)

        fname = os.path.join(save_dir, f"gradcam_{pred_label.lower()}_idx{idx}.png")
        plt.savefig(fname, dpi=130, bbox_inches="tight")
        plt.show()
        print(f"Saved → {fname}")


# ── Lead Importance Analysis ───────────────────────────────
def lead_importance(model, dataset, n_samples=200, save_dir="results/figures/"):
    """
    Permutation importance: zero out each lead and measure F1 drop.
    """
    from sklearn.metrics import f1_score
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # Collect samples
    xs, ys = [], []
    for i in range(min(n_samples, len(dataset))):
        x, y = dataset[i]
        xs.append(x); ys.append(y.item())
    xs = torch.stack(xs)   # (N, 12, 1000)
    ys = np.array(ys)

    # Baseline F1
    with torch.no_grad():
        preds = model(xs).argmax(1).numpy()
    baseline_f1 = f1_score(ys, preds, average="macro", zero_division=0)
    print(f"Baseline Macro F1: {baseline_f1:.4f}")

    # Permute each lead
    drops = []
    for lead_idx in range(12):
        x_perm = xs.clone()
        x_perm[:, lead_idx, :] = 0   # zero out this lead
        with torch.no_grad():
            perm_preds = model(x_perm).argmax(1).numpy()
        perm_f1 = f1_score(ys, perm_preds, average="macro", zero_division=0)
        drop    = baseline_f1 - perm_f1
        drops.append(drop)
        print(f"  Lead {LEAD_NAMES[lead_idx]:4s}: F1 drop = {drop:+.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["#ef4444" if d > 0 else "#10b981" for d in drops]
    bars    = ax.bar(LEAD_NAMES, drops, color=colors, edgecolor="white", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Lead Importance (Permutation)\nPositive = removing this lead hurts performance",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("ECG Lead")
    ax.set_ylabel("Macro F1 Drop")
    ax.spines[["top","right"]].set_visible(False)

    for bar, drop in zip(bars, drops):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.001,
                f"{drop:+.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    fname = os.path.join(save_dir, "lead_importance.png")
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"\nSaved → {fname}")
    return drops


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_DIR  = "data/ptbxl/records/"
    CKPT      = "checkpoints/cnn_transformer_best.pt"

    # Load model
    device = torch.device("cpu")
    model  = CNNTransformer(num_classes=3)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()
    print("Model loaded.")

    # Load test set
    test_ds = PTBXLDataset("data/test.csv", DATA_DIR, augment=False)
    print(f"Test samples: {len(test_ds)}")

    # 1. Grad-CAM — pick 1 sample per risk class
    inv_map = {v: k for k, v in LABEL_MAP.items()}
    indices = {}
    for i in range(len(test_ds)):
        _, y = test_ds[i]
        cls  = inv_map[y.item()]
        if cls not in indices:
            indices[cls] = i
        if len(indices) == 3:
            break

    print(f"\nRunning Grad-CAM on samples: {indices}")
    plot_gradcam(model, test_ds, list(indices.values()))

    # 2. Lead importance
    print("\nRunning Lead Importance Analysis...")
    lead_importance(model, test_ds, n_samples=300)