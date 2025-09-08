---

# Multi-Scale Attention Fusion Network with Anatomical Prior Learning (MAFN-APL)

This repository contains the PyTorch implementation of **MAFN-APL**, a novel deep learning architecture for **neuroimaging-based disease classification**

The model integrates **morphological priors**, **multi-scale attention**, **region-aware decomposition**, and **uncertainty-guided learning** to provide **robust predictions** with improved **interpretability**.

---

## 🚀 Key Contributions

* **Morphological Conv Block** – Mimics directional tumor/lesion growth patterns with anisotropic kernels and edge-preserving convolutions.
* **Uncertainty-Guided Attention** – Bayesian attention module that quantifies uncertainty using Monte Carlo dropout and reparameterization.
* **Multi-Scale Pyramid Attention** – Cross-scale communication via multi-head attention with hierarchical fusion.
* **Anatomical Region Decomposer** – Learnable masks for cortical, subcortical, and ventricular regions with region-specific encoders.
* **Cross-Region Attention Fusion** – Attention mechanism for integrating anatomical regions.
* **Temporal Consistency Module** – Self-supervised learning with augmented views for stable feature learning.
* **Composite Loss Function** – Combines classification, anatomical consistency, and uncertainty regularization losses.


## 🧠 Model Overview

The **MAFN-APL** consists of:

1. **Anatomical Region Decomposer**

   * Decomposes input into cortical, subcortical, and ventricular regions.
   * Each region passes through morphological + multi-scale encoders.

2. **Cross-Region Attention Fusion**

   * Learns interactions between anatomical regions with multi-head attention.

3. **Uncertainty-Guided Attention**

   * Bayesian attention weights with reparameterization trick.

4. **Temporal Consistency Module**

   * Enforces feature stability using augmented input versions.

5. **Adaptive Feature Aggregation + Classifier**

   * Global pooling, hierarchical fusion, and classification head.

---

## ▶️ Usage

### Model Inference

```python
import torch
from model import MAFN_APL

# Initialize model
model = MAFN_APL(num_classes=4, base_channels=64)

# Example input (batch of MRI scans, shape [B, C, H, W])
x = torch.randn(2, 3, 224, 224)

# Forward pass
outputs = model(x)

print("Output shape:", outputs.shape)  # [B, num_classes]
```

---

### Loss Function

```python
import torch
from loss import CompositeLoss
from model import MAFN_APL

model = MAFN_APL(num_classes=4)
criterion = CompositeLoss(num_classes=4)

x = torch.randn(4, 3, 224, 224)
y = torch.randint(0, 4, (4,))

outputs = model(x)
loss, loss_dict = criterion(outputs, y, model)

print("Total loss:", loss.item())
print("Loss breakdown:", loss_dict)
```

---

## 📊 Loss Components

1. **Classification Loss** → Standard cross-entropy.
2. **Anatomical Consistency Loss** → Enforces smoothness in learnable anatomical masks.
3. **Uncertainty Regularization** → Penalizes overconfident or unstable uncertainty estimates.

Final loss:

$$
\mathcal{L}_{total} = \mathcal{L}_{cls} + 0.1 \cdot \mathcal{L}_{anat} + 0.05 \cdot \mathcal{L}_{uncert}
$$

---




Would you like me to also **combine both projects (Alzheimer MoE + MAFN-APL)** into a **unified repo** with a single README comparing both models (baseline vs. novel architecture), or should each remain with its own README?
