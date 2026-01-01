# Visual Anomaly Detection on MVTec AD (Bottle)

An **unsupervised visual anomaly detection pipeline** implemented using a **Convolutional Autoencoder** and evaluated on the **MVTec Anomaly Detection (AD)** dataset.

The model is trained **exclusively on defect-free samples** and detects anomalies by measuring **reconstruction error**, which is visualized as **pixel-level heatmaps** highlighting defective regions.

---

## 📌 Overview

In real-world industrial inspection scenarios, defective samples are rare and highly diverse, making supervised learning approaches impractical. 

This project follows a **one-class (unsupervised) learning paradigm**:
* **Learn** the distribution of normal samples.
* **Identify** deviations from normality as anomalies.
* **Localize** defects at the pixel level.

The entire pipeline is modular, reproducible, and GPU-compatible.



---

## ✨ Key Features

* **Unsupervised Learning:** No defect labels required during training.
* **Deep Learning:** CNN-based autoencoder implemented in PyTorch.
* **Localization:** Pixel-level anomaly localization via reconstruction error.
* **Benchmarking:** Evaluation on the industry-standard MVTec AD dataset.
* **Performance:** Full GPU support (CUDA) for fast training and inference.

---

## 📂 Project Structure

```text
src/
 ├── datasets/
 │    └── mvtec_bottle.py      # Dataset loader for MVTec bottle category
 └── models/
      └── autoencoder.py       # Convolutional autoencoder model

scripts/
 ├── check_dataset.py          # Dataset sanity check
 ├── train_ae.py               # Autoencoder training (normal images only)
 └── infer_ae.py               # Inference and heatmap visualization

data/                          # Dataset directory (git-ignored)
runs/                          # Model outputs and visualizations (git-ignored)

```
## 📊 Dataset

This project uses the **MVTec Anomaly Detection (AD)** dataset, a widely adopted benchmark for
industrial visual anomaly detection.

Due to **dataset size and licensing restrictions**, the dataset is **not included** in this repository.

### Expected Directory Structure

```text
data/mvtec/bottle/
 ├── train/
 │    └── good/                # Normal samples used for training
 ├── test/
 │    ├── good/                # Normal test samples
 │    ├── broken_large/        # Defective samples
 │    ├── broken_small/        # Defective samples
 │    └── contamination/       # Defective samples
 └── ground_truth/             # Pixel-level defect masks
```
 Note: Ground-truth masks are used only for evaluation and visualization. They are never used during training.

## 🚀 Setup & Installation

This section describes how to set up the environment required to run the project locally.

---

### Virtual Environment

1.Create a Python virtual environment:

    python -m venv .venv

2. Activate the Environment

    Windows (PowerShell): .\.venv\Scripts\Activate.ps1

    Windows (CMD): .\.venv\Scripts\activate.bat

    Linux/macOS: source .venv/bin/activate


3. Install Dependencies

    pip install -r requirements.txt

4. GPU Support (Optional)

Verify if your NVIDIA GPU is available for PyTorch:
    python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

🛠 Usage

Step 1: Sanity Check

Ensure the dataset is correctly placed:

    python scripts/check_dataset.py

Step 2: Training

Train the autoencoder on "good" samples only:

    python scripts/train_ae.py

The trained model will be saved to runs/autoencoder_bottle.pth.

Step 3: Inference & Visualization

Generate anomaly heatmaps:

    python scripts/infer_ae.py

Output visualizations (original image + heatmap overlay) are saved in runs/infer/.


📝 Methodology Notes

    Training Strategy: Training is performed strictly on "normal" samples (train/good).

    Detection Mechanism: Anomalies are detected via pixel-wise reconstruction error. The logic is that the model, having only seen "good" bottles, will fail to accurately reconstruct "defective" parts.

    Exclusions: Dataset files, model weights, and output artifacts are excluded from version control for repository cleanliness.


🚀 Future Work

    [ ] Implement image-level and pixel-level AUROC evaluation.

    [ ] Add threshold-based anomaly decision mechanisms.

    [ ] Integrate feature-embedding approaches (e.g., ResNet-based methods).

    [ ] Extend to other MVTec categories (e.g., capsule, metal_nut).
