# Dysarthria / ALS Speech Detection

A Deep Learning pipeline to detect dysarthria and speech degradation (often caused by ALS or Cerebral Palsy) using the TORGO dataset.

## 🚀 Overview
This project implements a **Convolutional Recurrent Neural Network (CRNN)** to analyze audio patterns in speech. By combining CNNs for spectral feature extraction and BiLSTMs for temporal sequence modeling, the system achieves high sensitivity in detecting motor speech disorders.

## 📊 Performance Results
The model was evaluated using speaker-independent validation (no overlap of speakers between train and test sets).

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **90%** |
| **Dysarthric Recall** | **97%** |
| **Dysarthric Precision** | **76%** |
| **Normal Precision** | **99%** |

### Key Insights:
- **Temporal Awareness**: The BiLSTM architecture captures slurring and pauses over time, which are key indicators of dysarthria.
- **Clinical Reliability**: High recall (97%) ensures minimal false negatives, making it suitable for early screening.

## 🏗️ Architecture
- **Front-end**: 3-layer CNN for spectral feature extraction from Mel-Spectrograms.
- **Back-end**: 2-layer Bidirectional LSTM (BiLSTM) for temporal modeling.
- **Output**: Softmax classifier for Normal vs. Dysarthric detection.

## 🛠️ Setup & Usage

### 1. Requirements
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
The system uses the TORGO dataset. Run the parser to generate metadata:
```bash
python src/data/parse_torgo.py
```

### 3. Training
Train the CRNN model with Early Stopping:
```bash
python src/training/train.py
```

### 4. Evaluation
Generate diagnostic plots (Confusion Matrix, ROC, PR Curves):
```bash
python src/evaluation/evaluate.py
```

### 5. Inference
Predict dysarthria from a raw `.wav` file:
```bash
python inference.py path/to/audio.wav --threshold 0.7
```

## 📈 Visualizations
Diagnostic plots are saved in the `results/` directory:
- `confusion_matrix_norm.png`: Normalized detection performance.
- `training_curves.png`: Loss and accuracy history.
- `roc_curve.png`: Receiver Operating Characteristic.

## 📜 License
This project is for educational and research purposes. Data provided by the TORGO database.
