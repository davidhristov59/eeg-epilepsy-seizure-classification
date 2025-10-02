# 🧠 EEG Seizure Prediction using Feature Extraction and Machine Learning & Transformers

This project focuses on extracting meaningful features from EEG (electroencephalogram) signals for the purpose of **seizure detection and prediction**. It processes raw EEG data into time-domain, frequency-domain, and nonlinear features. It combines two complementary approaches: 

**1. Feature-based Machine Learning** – Extracting statistical, spectral, and nonlinear features from EEG signals and training classical ML classifiers.

**2. Deep Learning with Transformers** – Leveraging sequence modeling with Transformer architectures directly on preprocessed EEG signals for temporal pattern learning.

---

## ⚙️ Features Extracted

### Time-domain (Temporal) features
- Mean, variance, root mean square (RMS)
- Skewness and kurtosis
- Signal zero-crossings

### Frequency-domain (Spectral) features
- **Power Spectral Density (PSD)** using Welch's method
- Band power & ratios across:
  - Delta (0.5–4 Hz)
  - Theta (4–8 Hz)
  - Alpha (8–12 Hz)
  - Beta (12–30 Hz)
  - Gamma (30–100 Hz)
- Spectral entropy
- Peak, median, and total power

### Nonlinear (Complexity) features
- **Hjorth parameters**: activity, mobility, complexity
- **Fractal dimensions**:
  - Petrosian Fractal Dimension (PFD)
  - Higuchi Fractal Dimension (HFD)
- **Hurst exponent** (long-term memory of signal)

---

## 🤖 Models

### 1. Machine Learning ###
The extracted features are used to train and evaluate the following classifiers:

- **Support Vector Machine (SVM)**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Gradient Boosting (e.g., ADABoost)**
- Performance is evaluated using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

### 2. Deep Learning (Transformer Architecture)  

Instead of relying on hand-crafted features, we directly model the EEG sequences using a **Transformer Encoder**.  

####  Architecture Overview  
- **Input Representation**:  
  - EEG windows are represented as sequences of shape **(batch_size, sequence_length, n_channels)**.  
  - Each time step (multi-channel EEG vector) is projected into a **d_model** dimensional embedding via a learnable linear layer.  

- **Positional Encoding**:  
  - Since EEG data is sequential, sinusoidal positional encodings are added to embeddings to retain temporal order information.  

- **Transformer Encoder Layers**:  
  - Multiple stacked encoder layers (e.g., 2–6)  
  - Each encoder contains:  
    - **Multi-Head Self-Attention (MHSA)** to capture long-range dependencies between EEG time steps.  
    - **Feed-Forward Networks (FFN)** applied to each position.  
    - **Layer Normalization & Residual Connections** for stable training.  

- **Classification Head**:  
  - The output embeddings are pooled (mean/max/CLS token)  
  - A final **fully connected layer** maps to seizure vs. non-seizure classes.  

#### ⚙️ Training Setup  
- Optimizer: **AdamW** with weight decay  
- Scheduler: **Cosine Annealing / Step LR**  
- Loss: **Cross-Entropy Loss**  
- Regularization: Dropout + gradient clipping  

#### 🔍 Why Transformers for EEG?  
- Can **capture long-range temporal dependencies** across EEG signals (spikes, oscillations, preictal patterns).  
- Handles **multi-channel correlations** better than 1D-CNNs or RNNs.  
- Scales well with more data and patients.  

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
   git clone https://github.com/davidhristov59/eeg-epilepsy-seizure-classification.git
   cd eeg-epilepsy-seizure-classification

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
3. Preprocess the data
   ```bash
   python preprocessing.py

4. Merge the files into one file for training
   ```bash
   python merge.py

5. Run the classification pipeline for the models
   ```bash
   python classifiers.py

6. Visualize the results
   ```bash
   python plots_results.py

  ---

## 🧠 Dataset
[CHB-MIT EEG Dataset](https://physionet.org/content/chbmit/1.0.0/): Pediatric EEG dataset used for seizure detection publicly available on PhysioNet, from patients with epilepsy
