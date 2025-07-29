# 🧠 EEG Seizure Prediction using Signal Feature Extraction and Machine Learning

This project focuses on extracting meaningful features from EEG (electroencephalogram) signals for the purpose of **seizure detection and prediction**. It processes raw EEG data into time-domain, frequency-domain, and nonlinear features, which are then used to train and evaluate **machine learning models** for seizure classification.

---

## ⚙️ Features Extracted

### ✅ Time-domain (Temporal) features
- Mean, variance, root mean square (RMS)
- Skewness and kurtosis
- Signal zero-crossings

### ✅ Frequency-domain (Spectral) features
- **Power Spectral Density (PSD)** using Welch's method
- Band power & ratios across:
  - Delta (0.5–4 Hz)
  - Theta (4–8 Hz)
  - Alpha (8–12 Hz)
  - Beta (12–30 Hz)
  - Gamma (30–100 Hz)
- Spectral entropy
- Peak, median, and total power

### ✅ Nonlinear (Complexity) features
- **Hjorth parameters**: activity, mobility, complexity
- **Fractal dimensions**:
  - Petrosian Fractal Dimension (PFD)
  - Higuchi Fractal Dimension (HFD)
- **Hurst exponent** (long-term memory of signal)

---

## 🤖 Machine Learning Models

The extracted features are used to train and evaluate the following classifiers:

- **Support Vector Machine (SVM)**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Gradient Boosting (e.g., XGBoost)**
- Performance is evaluated using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

---

## 🧪 Methods Used

- **Sliding Window Technique**: Processes EEG signals in small overlapping windows (e.g., 1s with 50% overlap) to capture temporal patterns.
- **Fast Fourier Transform (FFT)**: Converts signal from time to frequency domain.
- **Welch's Method**: Estimates power spectrum using windowed FFT with averaging.
- **PyEEG Library**: Used for nonlinear EEG-specific features like Hjorth and fractal dimensions.
- **Scikit-learn**: For training and evaluating machine learning models.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/eeg-seizure-prediction.git
   cd eeg-seizure-prediction

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## 🧠 Dataset
[CHB-MIT EEG Dataset](https://physionet.org/content/chbmit/1.0.0/): Pediatric EEG dataset used for seizure detection publicly available on PhysioNet, from patients with epilepsy
