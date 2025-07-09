# Efficient3DNowcasting

This repository contains the code and experiments from my PhD work on efficient 3D convolutional neural networks for radar-based rainfall nowcasting.

## 📘 Project Overview

This study explores a variety of efficient 3D CNN architectures (e.g., Depthwise, Grouped, R(2+1)D, Ghost) and benchmarks their performance and resource usage for short-term rainfall prediction from radar sequences.

Key objectives:
- Benchmark efficient 3D convolutions against a standard U-Net baseline.
- Evaluate trade-offs between accuracy, latency, and parameter efficiency.
- Provide  robustness tests under data-scarce conditions.

- ## 🏗️ Project Structure
Efficient3DNowcasting/
│
├── models/ # 3D U-Net variants
├── training/ # Training pipelines
├── evaluation/ # Evaluation metrics & visualizations
├── data/ # Data loading or processing scripts
├── utils/ # Helper functions
├── README.md
├── requirements.txt
└── .gitignore


## 🧪 Requirements

- Python 3.9+
- PyTorch
- NumPy
- OpenCV
- tqdm

Install all dependencies:
```bash
pip install -r requirements.txt
