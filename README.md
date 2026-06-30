# Deepfake Detection & Computer Forensics Pipeline 🛡️🤖

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/Framework-PyTorch%20%2F%20TensorFlow-orange.svg)](https://pytorch.org/)
[![Interface](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the official implementation, data pipelines, and forensic reporting interface for the Computer Engineering Graduation Project titled **"Deepfake ile Manipüle Edilmiş Verinin Tespiti ve Sınıflandırılması"** at **Yozgat Bozok University**.

The project introduces a hybrid, multi-stream deep learning framework designed for Cyber Security Operations Centers (SOC) to detect hyper-realistic AI-generated video forgeries and preserve the digital chain of custody.

---

## 🌟 Key Features

- **Multi-Stream Architecture:** Integrates a spatial stream powered by **Xception / EfficientNet-B0** backbones with a frequency stream analyzing **FFT/DCT (Fast Fourier Transform)** spectral artifacts left by GANs.
- **Data Leakage Prevention Engine:** Utilizes a custom video ID-based chronological split algorithm rather than frame-based random distribution to ensure empirical validity.
- **Explainable AI (XAI):** Implements **Grad-CAM** activation maps to neutralize the "black-box" dilemma of deep neural networks, pinpointing exactly where modifications occur (eyes, mouth, skin textures).
- **Forensic-Grade SOC Interface:** An asynchronous **Streamlit** user interface featuring real-time bounding box tracking, automated forensic log generation ($\mu s$ resolution), and a static analysis panel.
- **State-of-the-Art Performance:** Achieved **97% Overall Accuracy**, **99% Precision** for fake content verification, and a stable **0.97 F1-Score** under aggressive network compression and JPEG constraints.

---

## 📐 System Architecture & Workflow

The pipeline ingests raw videos and executes the following operations systematically:
1. **Preprocessing:** Face detection and alignment via MTCNN, followed by chronological data splitting.
2. **Augmentation:** Robustness enhancement against real-world noise using the Albumentations library.
3. **Feature Fusion:** Concatenation of spatial convolutional features and log-transformed magnitude spectrums.
4. **Inference & Explanation:** Probabilistic risk scoring ($P_{fake} \ge 0.50$) backed by local pixel localization (Grad-CAM).

---

## 🛠️ Installation & Setup

### Prerequisites
Ensure you have Python 3.9+ and a CUDA-supported GPU environment.

```bash
# Clone the repository
git clone [https://github.com/m-enesbaysal/bozok-deepfake.git](https://github.com/m-enesbaysal/bozok-deepfake.git)
cd bozok-deepfake

# Install dependencies
pip install -r requirements.txt


To boot up the Streamlit-driven analytics dashboard:
streamlit run src/interface.py
