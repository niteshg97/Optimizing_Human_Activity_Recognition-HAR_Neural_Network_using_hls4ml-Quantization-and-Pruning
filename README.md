# 🧠 Human Activity Recognition (HAR) — Model Compression & FPGA Deployment using hls4ml 🚀  

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![hls4ml](https://img.shields.io/badge/hls4ml-FPGA--Optimized-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🏗️ Project Overview  

This project presents a **novel approach to model compression and FPGA deployment** for a **Human Activity Recognition (HAR)** model using **quantization**, **pruning**, and **decomposition techniques**. This project optimizes a Keras HAR neural
network using 16-bit quantization and 80% pruning. The result is a low-latency, low-power design ideal
for real-time edge AI, with applications in wearable fitness trackers, real-time fall detection, and
industrial IoT monitoring.

The workflow combines **TensorFlow**, **TensorFlow Model Optimization Toolkit (TF-MOT)**, and **hls4ml** to generate FPGA-ready hardware implementations while achieving an optimal **performance-efficiency trade-off**.  

---

## 🎯 Objectives  

✅ Develop a lightweight, high-accuracy ML model for HAR  
✅ Integrate **quantization**, **pruning**, and **model decomposition** techniques  
✅ Achieve **FPGA synthesis** using `hls4ml` for hardware efficiency  
✅ Analyze **accuracy vs. latency vs. resource utilization** trade-offs  

---

## ⚙️ Tools & Frameworks Used  

| Tool / Framework | Purpose |
|------------------|----------|
| 🧩 **TensorFlow / Keras** | Model training and evaluation |
| ✂️ **TensorFlow Model Optimization Toolkit** | Model pruning & quantization |
| ⚡ **hls4ml** | High-Level Synthesis for FPGA deployment |
| 📊 **scikit-learn & NumPy** | Data preprocessing and evaluation |
| 💾 **Vivado HLS / Vitis HLS (optional)** | FPGA synthesis and simulation |
| ☁️ **Google Colab** | Training and HLS conversion environment |

---

## 🧠 Model Architecture  

| Layer | Type | Output Shape | Activation |
|--------|------|---------------|-------------|
| Input | Dense | (561,) | — |
| Dense-1 | Fully Connected | 64 | ReLU |
| Dense-2 | Fully Connected | 32 | ReLU |
| Output | Fully Connected | 6 | Softmax |

**Dataset:** UCI Human Activity Recognition (HAR)  
**Classes:** WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING  

---

## 🧮 Methodology  

### 🔹 Phase 1 — Baseline Training  
- Trained the model using Keras on the UCI HAR dataset  
- Achieved **95.18% accuracy** on test data  

### 🔹 Phase 2 — Quantization  
- Converted model to **fixed-point precision (`ap_fixed<16,6>`)**  
- Verified minimal loss in accuracy  

### 🔹 Phase 3 — Pruning & Fine-Tuning  
- Applied **80% pruning** using TensorFlow MOT  
- Fine-tuned model to recover accuracy  

### 🔹 Phase 4 — hls4ml Conversion  
- Converted the optimized model into an FPGA-compatible HLS project  
- Compiled and tested **C-simulation** successfully  

---

## 📊 Performance Summary  

| Configuration | Reuse Factor | Accuracy | Latency | Resource Usage |
|----------------|--------------|-----------|----------|----------------|
| Baseline Keras | — | **95.18%** | — | — |
| Quantized Model | 1 | **93.9%** | Low | High |
| Pruned (80%) | 1 | **93.5%** | Low | Very Low |
| Serialized (RF=51) | 51 | **93.9%** | High | Lowest |

---

## ⚡ FPGA Configuration  

| Parameter | Value |
|------------|--------|
| Target FPGA | `xc7k160t-fbg484-1` |
| Clock Period | `5 ns` |
| IO Type | `io_parallel` |
| Precision | `ap_fixed<16,6>` |
| Reuse Factor | `1` and `51` (tested) |

---

## 📁 Repository Structure  

```bash
.
├── Optimizing_a_HAR_Neural_Network.ipynb     # Google Colab notebook (main project)
├── har_model.h5                              # Baseline trained model
├── har_model_pruned.h5                       # 80% pruned fine-tuned model
├── HAR_HLS4ML_Project.zip                    # hls4ml-generated FPGA project files
└── README.md                                 # This file




---
```
## 🚀 Quick Start (Google Colab)

```bash

# Install dependencies
!pip install tensorflow tensorflow-model-optimization hls4ml scikit-learn numpy

# Run the notebook
execute Optimizing_a_HAR_Neural_Network.ipynb


```

## 🧩 Key Learnings  

- ⚙️ **Fixed-point quantization** can significantly reduce FPGA latency and area with minimal accuracy drop.  
- ✂️ **Structured pruning (up to 80%)** improves synthesis efficiency without heavy accuracy degradation.  
- 🧮 **Removing or linearizing Softmax** before `hls4ml` conversion prevents C++ compile issues (`implementation` field mismatch).  
- 🤝 **Combining compression and hardware co-design** yields real-time performance on embedded devices.  



## 🙌 Acknowledgements  

- [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)  
- [hls4ml — FastMachineLearning Project](https://github.com/fastmachinelearning/hls4ml)  
- [TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization)  
- [Google Colab](https://colab.research.google.com)  

## 👨‍💻 Author  

**Nitesh Kumar**  
📧 niteshk.ug23.ee@nitp.ac.in  
🌐 [LinkedIn Profile](https://www.linkedin.com/in/nitesh-kumar-68a698275)  




## 🛡️ License  

**MIT License**  
Copyright (c) 2025 Nitesh Kumar







...

