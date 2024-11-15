import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define data paths to load embeddings stored
data_normal = 'data/embedings/normal' # path to embedding folder for non-Dysarthic speakers
data_dys_mild = 'data/embedings/mild'
data_dys_moderate = 'data/embedings/moderate'
data_dys_severe = 'data/embedings/severe'


def load_embeddings_and_labels(data_paths, label_values):
    """Loads embeddings and corresponding labels from multiple directories."""
    embeddings = []
    labels = []

    for path, label in zip(data_paths, label_values):
        # Iterate through each file in the directory
        for filename in os.listdir(path):
            if filename.endswith(".npy"):  # Assuming embeddings are stored as .npy files
                # Load the embedding
                embedding = np.load(os.path.join(path, filename))

                # Check if the embedding is empty. If so, skip it
                if embedding.size == 0:
                    print(f"Warning: Empty embedding found in file: {filename}. Skipping.")
                    continue  

                embeddings.append(embedding)
                labels.append(label)

    # Convert embeddings and labels to NumPy arrays
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    return embeddings, labels

def perform_svm_classification(X, y, title):
    """Performs SVM classification and prints the accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    svm_classifier = SVC(kernel='rbf')
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{title}: Accuracy = {accuracy:.4f}")

# Severity classification
embeddings, labels = load_embeddings_and_labels([data_dys_mild, data_dys_moderate, data_dys_severe], [1, 2, 3])
perform_svm_classification(embeddings, labels, "Severity Classification")

# Binary classification 
embeddings, labels = load_embeddings_and_labels([data_normal, data_dys_mild, data_dys_moderate, data_dys_severe], [0, 1, 1, 1])
perform_svm_classification(embeddings, labels, "Binary Classification")
