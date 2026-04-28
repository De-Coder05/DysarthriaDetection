# Dysarthria / ALS Speech Detection

A Deep Learning pipeline to detect dysarthria and speech degradation (often caused by ALS or Cerebral Palsy) using the TORGO dataset.

## 🚀 Overview
This project implements a **Convolutional Recurrent Neural Network (CRNN)** to analyze audio patterns in speech. By combining CNNs for spectral feature extraction and BiLSTMs for temporal sequence modeling, the system achieves high sensitivity in detecting motor speech disorders.

## 📊 Performance Results
The model achieves high precision across all diagnostic categories:

| Severity | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Normal** | 0.99 | **99%** | 0.99 |
| **Mild** | 0.96 | **98%** | 0.97 |
| **Moderate** | 0.99 | **96%** | 0.98 |
| **Severe** | 0.97 | **96%** | 0.97 |
| **Accuracy** | | | **98%** |

### Key Insights:
- **Diagnostic Precision**: The model doesn't just detect dysarthria; it accurately grades the severity (Mild to Severe).
- **Temporal Awareness**: The BiLSTM architecture captures slurring and pauses over time, which are key indicators of ALS-related speech degradation.

## 🏗️ Architecture
- **Front-end**: 3-layer CNN for spectral feature extraction from Mel-Spectrograms.
- **Back-end**: 2-layer Bidirectional LSTM (BiLSTM) for temporal modeling.
- **Output**: 4-class Softmax (Normal, Mild, Moderate, Severe).

## 🎯 Quick Start (Inference)
The repository includes a pre-trained model (`models/best_model.pth`), so you can run predictions immediately!

**1. Clone the repository and install requirements**
```bash
git clone https://github.com/De-Coder05/DysarthriaDetection.git
cd DysarthriaDetection
pip install -r requirements.txt
```

**2. Run inference on your own audio file**
```bash
python inference.py path/to/your/audio_file.wav
```

## 🛠️ Setup & Usage

### 1. Requirements
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
The system uses the TORGO dataset. Run the parser to generate metadata with severity labels:
```bash
python src/data/parse_torgo.py
```

### 3. Training
Train the 4-class CRNN model with Stratified Splitting and Early Stopping:
```bash
python src/training/train.py
```

### 4. Evaluation
Generate 4x4 diagnostic confusion matrices and reports:
```bash
python src/evaluation/evaluate.py
```

### 5. Inference
Predict severity from a raw `.wav` file:
```bash
python inference.py path/to/audio.wav
```

## 📈 Visualizations
Diagnostic plots are saved in the `results/` directory:
- `confusion_matrix_norm.png`: 4x4 Normalized diagnostic performance.
- `training_curves.png`: Loss and accuracy history.

## 📜 License
This project is for educational and research purposes. Data provided by the TORGO database.
