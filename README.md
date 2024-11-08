# Dysarthria Classification Using Speech Embeddings from Whisper and SVM

This repository contains code and resources for classifying **Dysarthria** severity levels using **Speech Embeddings** extracted from OpenAI's **Whisper** model. We use **Support Vector Machines (SVM)** for classification based on embeddings obtained from the **Torgo** and **UA Speech** datasets.

The goal is to classify speech samples into two tasks:
1. **Normal vs. Dysarthria** (binary classification)
2. **Mild, Moderate, Severe Dysarthria** (multiclass classification)

## Project Overview

- **Datasets**: The data is sourced from the **Torgo** dataset and **UA Speech** dataset. The speech samples are divided into three categories: **Normal**, **Mild**, **Moderate**, and **Severe** levels of dysarthria.
  
- **Model Used**: The **Whisper model** is used to extract high-dimensional speech embeddings that capture important acoustic features. These embeddings are then used as input to an **SVM classifier**.

- **Classification**: The classification tasks are carried out using **SVM** (Support Vector Machines), a widely used machine learning algorithm for classification tasks.

## Requirements

Before running the code, make sure to install the required dependencies. You can use the provided `requirements.txt` file.

### Dependencies:

- `whisper` – For speech feature extraction using Whisper model.
- `torch` – PyTorch framework for model inference.
- `librosa` – Audio loading and preprocessing.
- `numpy` – Numerical handling of embeddings.
- `scikit-learn` – For training the SVM classifier.

You can install these dependencies with:

```bash
pip install -r requirements.txt
```

Here’s the `requirements.txt` file:

```
git+https://github.com/openai/whisper.git  # Whisper package from GitHub
torch>=1.10.0                          # PyTorch (required for running Whisper)
librosa>=0.8.1                         # For audio preprocessing (loading and resampling)
numpy>=1.21.0                          # For numerical operations
scikit-learn>=0.24.2                    # For machine learning (e.g., SVM classifier)
```

## Directory Structure

The directory structure follows the format below:

```
dysarthria-classification/
│
├── torgo/                         # Folder containing Torgo dataset
│   ├── audio/                     # Audio files organized by class and speaker
│   │   ├── mild/                  # Mild Dysarthria audio files
│   │   ├── moderate/              # Moderate Dysarthria audio files
│   │   └── severe/                # Severe Dysarthria audio files
│   └── embeddings/                # Extracted Whisper embeddings (in .npy format)
│       ├── mild/                  # Embeddings for mild class
│       ├── moderate/              # Embeddings for moderate class
│       └── severe/                # Embeddings for severe class
│
├── classify_dysarthria.py         # Script for training and evaluating SVM models
└── README.md                      # This README file
```

## Step-by-Step Guide

### 1. **Data Organization**

The speech audio data is organized in the `torgo/audio/class/speakername/audio.wav` format, and the Whisper embeddings are stored as `.npy` files in the `torgo/embeddings/class/embedding_file.npy` directory.

- **Audio Files**: 
  - `torgo/audio/mild/speaker1/audio.wav`
  - `torgo/audio/moderate/speaker2/audio.wav`
  - `torgo/audio/severe/speaker3/audio.wav`

- **Embeddings**:
  - `torgo/embeddings/mild/embedding_speaker1.npy`
  - `torgo/embeddings/moderate/embedding_speaker2.npy`
  - `torgo/embeddings/severe/embedding_speaker3.npy`

### 2. **Extracting Embeddings Using Whisper**

The Whisper model is used to extract embeddings from the audio files. You can modify the code to extract embeddings if you don’t have them yet. Here’s a sample function to [https://github.com/sugarcane-mk/whisper/blob/main/extract_embeddings.py](extract embeddings) from audio using Whisper

### 3. **Training the SVM Classifier**

Once the embeddings are extracted, you can train an SVM classifier to perform the dysarthria classification. The code below trains an SVM to classify speech as **Normal vs. Dysarthria** or **Mild, Moderate, Severe**:

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# Load embeddings and labels
def load_embeddings_and_labels(embedding_dir, label):
    embeddings = []
    labels = []
    
    for filename in os.listdir(embedding_dir):
        if filename.endswith(".npy"):
            embedding = np.load(os.path.join(embedding_dir, filename))
            embeddings.append(embedding)
            labels.append(label)
    
    return np.array(embeddings), np.array(labels)

# Example: Load data for binary classification (Normal vs Dysarthria)
mild_embeddings, mild_labels = load_embeddings_and_labels('torgo/embeddings/mild/', 0)  # 0 for Dysarthria
normal_embeddings, normal_labels = load_embeddings_and_labels('torgo/embeddings/normal/', 1)  # 1 for Normal

# Combine datasets
X = np.vstack([mild_embeddings, normal_embeddings])
y = np.hstack([mild_labels, normal_labels])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM classifier
svm_clf = SVC(kernel='linear')
svm_clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred = svm_clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

### 4. **Evaluating the Model**

You can evaluate the trained SVM model on the test set and generate classification metrics using the `classification_report` function from `scikit-learn`.

### 5. **Running the Script**

To run the classification process, you can execute the `classify_dysarthria.py` script which will:
- Load the extracted embeddings.
- Train the SVM classifier.
- Evaluate performance on the test set.

```bash
python classify_dysarthria.py
```

## Example Outputs

After running the classification script, you’ll get an output like this:

```
Classification Report:
              precision    recall  f1-score   support

        0.0       0.80      0.85      0.82        50
        1.0       0.83      0.79      0.81        50

    accuracy                           0.81       100
   macro avg       0.81      0.82      0.81       100
weighted avg       0.81      0.81      0.81       100
```

## Conclusion

This repository provides a pipeline for classifying **Dysarthria** levels using speech embeddings extracted by the **Whisper model** and **SVM**. You can easily modify the dataset and classification tasks to fit your specific needs, such as exploring additional models or fine-tuning Whisper for even better performance.

## License

This repository is open-source and available under the **MIT License**.

---

Let me know if you need further modifications or more details!
