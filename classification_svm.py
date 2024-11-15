# Load Libs
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.decomposition import PCA

# Define functions

# Load embedings and assign lables
def load_embeddings_and_labels(data_dirs,label):
    embeddings = []
    labels = []
    for path, label in zip(data_paths, labels):
        for filename in os.listdir(path):
            if filename.endswith(".npy"):
                embedding = np.load(os.path.join(path, filename))
                embeddings.append(embedding)
                labels.append(label)
    return np.array(embeddings), np.array(labels)

# SVM classifier
def perform_svm_classification(X, y, title):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Split based on class lables
    svm_classifier = SVC(kernel='rbf')
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy ({title}): {accuracy}")
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, title)
    # plot_decision_boundary(svm_classifier, X_train, y_train, title)

# Confusion matrix Plot
def plot_confusion_matrix(y_true, y_pred, title):
  cm = confusion_matrix(y_true, y_pred)
  print("Confusion Matrix:")
  print(cm)
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.title(f'Confusion Matrix ({title})')
  plt.show()

# PCA
def plot_decision_boundary(classifier, X, y, title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    h = .02
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = classifier.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.5)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolor='k')
    plt.title(f'Decision Boundary ({title})')
    plt.show()

# View clasiification for each test data
def svm_classification_results(X, y, title):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm_classifier = SVC(kernel='rbf')
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)

    # Print predicted and actual classes for each embedding in the test set
    for i in range(len(y_test)):
        print(f"Embedding {i+1}: Predicted Class - {y_pred[i]}, Actual Class - {y_test[i]}")

# Analyse class distribution across data
def class_distribution(labels):
    class_counts = Counter(labels)
    total_samples = len(labels)
    for label, count in class_counts.items():
        percentage = (count / total_samples) * 100
        print(f"Class {label}: {count} samples ({percentage:.2f}%)")

# Example usage (replace with your actual labels):
# Assuming 'labels' is a NumPy array of your class labels

# Analysing train and test data
def svm_classification_analysis(X, y, title):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Random samplings
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Stratified samplings

    # Print the size of train and test data for each class
    print(f"Data sizes for {title}:")
    for class_label in np.unique(y):
        train_count = np.sum(y_train == class_label)
        test_count = np.sum(y_test == class_label)
        print(f"Class {class_label}: Train - {train_count}, Test - {test_count}")

# Sample usage to call functions

# Define paths to embeddings file
data_control='/content/drive/UAspeech/whisper_small/control'
data_verylow='/content/drive/UAspeech/whisper_small/verylow'
data_low='/content/drive/UAspeech/whisper_small/low'
data_medium='/content/drive/UAspeech/whisper_small/mid'
data_high='/content/drive/UAspeech/whisper_small/high'

# Severity classification
data_paths = [data_verylow, data_low, data_medium, data_high]
labels = ['verylow', 'low', 'medium', 'high']

# Load embeddings and labels
embeddings_severe, lables_severe = load_embeddings_and_labels(data_paths,labels)
print(embeddings_severe.shape) # (data size, hidden state)
print(lables_severe.shape)  # (label_size,)
print(np.unique(lables_severe))

# Call Svm classifier
perform_svm_classification(embeddings_severe, lables_severe, "Whisper_small")

# Binary classification
data_paths = [data_control, data_verylow, data_low, data_medium, data_high]
labels = ['Control', 'Dysarthria', 'Dysarthria', 'Dysarthria', 'Dysarthria']

# Load embeddings and labels
embeddings_binary, lables_binary = load_embeddings_and_labels(data_paths,labels)
print(embeddings_binary.shape) # (data size, hidden state)
print(lables_binary.shape)  # (label_size,)
print(np.unique(lables_binary))

# Call Svm classifier
perform_svm_classification(embeddings_binary, lables_binary, "Whisper_small")
