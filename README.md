# BrainBeat

### Live Demo: [Open BrainBeat](https://brainbeat.streamlit.app/)

## Multimodal EEG–ECG Seizure Onset Detection Using Decision-Level Fusion

BrainBeat is a multimodal machine learning system designed for seizure onset detection using two physiological signal modalities: EEG (Electroencephalography) and ECG (Electrocardiography).

EEG captures electrical activity of the brain, while ECG captures electrical activity of the heart. BrainBeat independently processes both signals, extracts relevant features, applies machine learning models, and combines their predictions using decision-level fusion.

The project focuses on building a modular machine learning pipeline for biomedical signal analysis and demonstrating how multiple physiological modalities can be combined for seizure detection.

---

## Project Overview

The system consists of two independent processing pipelines:

1. EEG-based seizure detection
2. ECG-based abnormality detection

The predictions from both pipelines are then combined using an OR-based decision-level fusion strategy.

```text
                    EEG Signal
                         |
                         v
                  5-Second Windows
                         |
                         v
                  Feature Extraction
                         |
                         v
                   KNN Classifier
                         |
                         v
                   EEG Prediction
                         |
                         |
                         v
              +----------------------+
              | Decision-Level Fusion|
              |       EEG OR ECG     |
              +----------------------+
                         ^
                         |
                   ECG Prediction
                         ^
                         |
                   KNN Classifier
                         ^
                         |
                  Feature Extraction
                         ^
                         |
                  10-Second Windows
                         ^
                         |
                    ECG Signal
```
The final fusion rule is:


Final Decision = EEG Decision OR ECG Decision


If either modality produces a positive/abnormal prediction, the system generates a seizure alert.

---

## Objectives

* Develop an EEG-based seizure detection pipeline.
* Develop an ECG-based abnormality detection pipeline.
* Process physiological signals into machine learning-compatible features.
* Compare multiple machine learning classification algorithms.
* Select the best-performing model for each modality.
* Combine EEG and ECG predictions using decision-level fusion.
* Visualize physiological signals through a user interface.
* Evaluate the models using multiple classification metrics.

---

## Datasets

### CHB-MIT Scalp EEG Database

The EEG component uses the CHB-MIT Scalp EEG Database available through PhysioNet.

The recordings are provided in EDF format and are processed using MNE.

Dataset:

[https://physionet.org/content/chbmit/](https://physionet.org/content/chbmit/)

### MIT-BIH Arrhythmia Database

The ECG component uses the MIT-BIH Arrhythmia Database available through PhysioNet.

The recordings are loaded and processed using WFDB and NeuroKit2.

Dataset:

[https://physionet.org/content/mitdb/](https://physionet.org/content/mitdb/)

---

## EEG Processing

The EEG pipeline processes the recordings into fixed-length windows and extracts statistical and frequency-domain features.

```text
EDF EEG File
     |
     v
MNE
     |
     v
5-Second Windowing
     |
     v
Feature Extraction
     |
     +--> Mean
     +--> Variance
     +--> Skewness
     +--> Kurtosis
     +--> Frequency/Band Energy Features
     |
     v
Machine Learning Classifier
     |
     v
EEG Prediction
```

### EEG Windowing

The continuous EEG recordings are divided into 5-second windows.

Each window is treated as an individual machine learning sample.

### EEG Features

The following types of features are extracted:

* Mean
* Variance
* Skewness
* Kurtosis
* Frequency-domain features
* Band-specific energy features

### EEG Models

The following machine learning models were evaluated:

* Logistic Regression
* Support Vector Machine with Linear Kernel
* Support Vector Machine with RBF Kernel
* Random Forest
* K-Nearest Neighbors

---

## ECG Processing

The ECG pipeline processes the recordings using windowing, R-peak detection, RR interval analysis, and HRV feature extraction.

```text
ECG Record
     |
     v
WFDB
     |
     v
10-Second Windowing
     |
     v
NeuroKit2
     |
     v
R-Peak Detection
     |
     v
RR Interval Calculation
     |
     v
HRV Feature Extraction
     |
     +--> Mean RR
     +--> SDNN
     |
     v
Machine Learning Classifier
     |
     v
ECG Prediction
```

### ECG Windowing

The ECG recordings are divided into 10-second windows.

A longer window is used for ECG because multiple heartbeats are required for extracting RR interval and HRV-related information.

### R-Peak Detection

R-peaks represent the prominent peaks in the ECG waveform associated with individual heartbeats.

The time difference between consecutive R-peaks is used to calculate RR intervals.

### ECG Features

The current implementation uses features including:

* Mean RR interval
* SDNN
* RR interval / HRV information

### ECG Models

The following machine learning models were evaluated:

* Logistic Regression
* Support Vector Machine with RBF Kernel
* Random Forest
* K-Nearest Neighbors

---

## Machine Learning Models

BrainBeat evaluates multiple machine learning algorithms before selecting the final classifier.

The selected model for both EEG and ECG is K-Nearest Neighbors (KNN) with:

```text
k = 5
```

KNN classifies a new sample by finding the nearest samples in the feature space and assigning a class based on their labels.

---

## Evaluation Metrics

BrainBeat evaluates model performance using multiple classification metrics.

### Accuracy

Accuracy represents the proportion of correctly classified samples out of all samples.

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Precision

Precision measures how many of the samples predicted as positive were actually positive.

```text
Precision = TP / (TP + FP)
```

### Recall

Recall measures how many of the actual positive samples were correctly detected.

```text
Recall = TP / (TP + FN)
```

### F1-Score

F1-score provides a balance between precision and recall.

```text
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

F1-score is particularly important in this project because biomedical datasets can contain class imbalance, making accuracy alone insufficient for evaluating classification performance.

---

## EEG Model Evaluation

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   95.40% |      0.41 |   0.28 |     0.33 |
| SVM Linear          |   88.20% |      0.36 |   0.52 |     0.42 |
| SVM RBF             |   99.35% |      0.74 |   0.96 |     0.84 |
| Random Forest       |   90.10% |      0.39 |   0.31 |     0.34 |
| KNN (k=5)           |   99.60% |      0.92 |   0.84 |     0.88 |

KNN achieved the best overall balance among the evaluated EEG models, with an accuracy of 99.60% and an F1-score of 0.88.

Stratified five-fold cross-validation was used for EEG evaluation.

---

## ECG Model Evaluation

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   70.00% |      0.59 |   0.71 |     0.64 |
| SVM RBF             |   66.00% |      0.53 |   1.00 |     0.69 |
| Random Forest       |   91.00% |      0.94 |   0.83 |     0.88 |
| KNN (k=5)           |   91.70% |      0.92 |   0.85 |     0.89 |

KNN achieved the highest accuracy among the evaluated ECG models, with an accuracy of 91.70% and an F1-score of 0.89.

A stratified train-test split was used for ECG evaluation.

---

## Decision-Level Fusion

After independently processing the EEG and ECG signals, BrainBeat combines their final predictions using decision-level fusion.

The implemented fusion rule is:

```text
Final Decision = EEG OR ECG
```

### Fusion Logic

| EEG Prediction | ECG Prediction | Final Decision |
| -------------- | -------------- | -------------- |
| Normal         | Normal         | Normal         |
| Seizure        | Normal         | Seizure Alert  |
| Normal         | Abnormal       | Seizure Alert  |
| Seizure        | Abnormal       | Seizure Alert  |

The OR-based strategy prioritizes detecting potentially abnormal events.

However, because a positive prediction from either modality can trigger the final alert, this approach can also increase false-positive predictions.

---

## User Interface

BrainBeat uses Streamlit to provide a lightweight interface for interacting with the system.

The interface provides functionality for:

* EEG input
* ECG input
* Signal visualization
* EEG prediction
* ECG prediction
* Final fused prediction

The interface allows the complete processing pipeline to be accessed without manually executing each processing stage.

---

## Technology Stack

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Logistic Regression
* Support Vector Machine
* Random Forest
* K-Nearest Neighbors

### EEG Processing

* MNE
* NumPy
* Pandas

### ECG Processing

* WFDB
* NeuroKit2
* NumPy
* Pandas

### Visualization

* Matplotlib

### User Interface

* Streamlit

### Development Tools

* Jupyter Notebook
* Google Colab
* Visual Studio Code

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/shubhankar05sarkar/BrainBeat.git
```

Navigate to the project directory:

```bash
cd BrainBeat
```

### 2. Create a Virtual Environment

```powershell
python -m venv venv
```

### 3. Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\activate
```

After successful activation, the terminal should display:

```text
(venv)
```

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## Running the Project

Run the project from the root directory of the repository.

Use the following commands:

```powershell
.\venv\Scripts\activate
streamlit run frontend/app.py
```

Alternatively, if the virtual environment is already activated:

```powershell
streamlit run frontend/app.py
```

After running the command, Streamlit will start the application and provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in a web browser to access the BrainBeat interface.

### Important

Run the Streamlit command from the project root directory:

```text
seizure_onset_detection/
```

---

## Performance Summary

| Modality | Selected Model | Accuracy | Precision | Recall | F1-Score |
| -------- | -------------- | -------: | --------: | -----: | -------: |
| EEG      | KNN (k=5)      |   99.60% |      0.92 |   0.84 |     0.88 |
| ECG      | KNN (k=5)      |   91.70% |      0.92 |   0.85 |     0.89 |

---

## Limitations

* The EEG and ECG datasets are obtained from independent databases rather than synchronized EEG-ECG recordings from the same patients.
* The current OR-based fusion strategy can increase false-positive alerts.
* Post-fusion performance requires further quantitative evaluation.
* The current implementation is a research prototype and has not been clinically validated.
* More extensive EEG preprocessing and artifact removal can be explored.

---

## Future Scope

* Use synchronized EEG-ECG recordings from the same patients.
* Evaluate patient-specific and cross-patient generalization.
* Investigate weighted decision-level fusion.
* Compare decision-level fusion with feature-level and hybrid fusion.
* Explore CNN, LSTM, CNN-LSTM, and Transformer-based models.
* Improve EEG artifact and noise removal.
* Incorporate additional HRV features.
* Explore real-time EEG and ECG monitoring.
* Investigate wearable-device integration.

---

## Disclaimer

BrainBeat is an academic research prototype developed for educational and research purposes.

It is not intended to provide medical diagnosis or replace evaluation by qualified healthcare professionals.

The reported results are based on experiments performed on publicly available datasets and should not be interpreted as clinical validation.
