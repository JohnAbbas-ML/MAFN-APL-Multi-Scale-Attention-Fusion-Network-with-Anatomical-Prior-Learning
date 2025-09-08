---

# 🧠 MAFN-APL: Multi-Scale Attention Fusion Network with Anatomical Prior Learning

This repository provides an **end-to-end PyTorch implementation** of the **MAFN-APL model** for **brain tumor classification**.
The model introduces several **novel architectural components** and a **composite loss function** tailored for medical imaging tasks.

---

## 🚀 Key Features

### 🏗️ Novel Architectural Components

* **Morphological Convolution Block**
  Mimics **tumor growth patterns** with directional convolutions & edge-preserving operators.

* **Uncertainty-Guided Attention (UGA)**
  Bayesian attention with Monte Carlo dropout to **quantify predictive uncertainty**.

* **Multi-Scale Pyramid Attention (MSPA)**
  Extracts multi-resolution features and enables **cross-scale communication**.

* **Anatomical Region Decomposer (ARD)**
  Learns **cortical, subcortical, and ventricular masks** with region-specific encoders.

* **Cross-Region Attention Fusion (CRAF)**
  Models **inter-region dependencies** via multi-head attention fusion.

* **Temporal Consistency Module (TCM)**
  Enforces **self-supervised consistency** between augmented feature representations.

---

### 🎯 Composite Loss Function

The **`CompositeLoss`** combines three objectives:

1. **Cross-Entropy Loss** → for classification
2. **Anatomical Consistency Loss** → enforces smooth anatomical masks
3. **Uncertainty Regularization** → penalizes unstable attention distributions

Final Loss:

$$
L_{total} = L_{cls} + 0.1 \cdot L_{anat} + 0.05 \cdot L_{uncertainty}
$$

## 🧪 Research Contribution

The **MAFN-APL** framework is designed for **robust brain tumor classification** with:

* **Anatomical priors** to guide region-specific learning.
* **Attention with uncertainty** for interpretable predictions.
* **Consistency learning** for better generalization.

This makes it well-suited for **MRI-based medical diagnosis**.
