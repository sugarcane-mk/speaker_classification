# Audio Embedding Extraction

This repository contains a Python script to extract embeddings from the OpenAI Whisper model for speech classification or other downstream tasks. Whisper is a powerful automatic speech recognition (ASR) model that can be used to generate speech embeddings from raw audio input.

## Requirements

To run the code, you will need to install the following Python packages:

- `whisper` - For loading the Whisper model.
- `torch` - For PyTorch model inference.
- `numpy` - For numerical operations and handling embeddings.
- `librosa` - For audio preprocessing (if you want to preprocess audio files before using the Whisper model).

### Installation

You can install the required libraries using pip:

```bash
pip install git+https://github.com/openai/whisper.git
pip install torch numpy librosa
```

## Usage

### Step 1: Prepare Audio File

Ensure that your audio file is in a standard format, such as WAV, MP3, or FLAC. The audio should be in **mono** and sampled at **16kHz**, which is the format Whisper expects.

### Step 2: Extract Embeddings

Run the provided Python script to extract embeddings from the audio file. The embeddings are the high-dimensional features generated by Whisper’s encoder, which you can use for downstream tasks such as speech classification or clustering.

### Example Code

```python
import whisper
import numpy as np

# Load the Whisper model (you can choose any model size like "small", "medium", "large")
model = whisper.load_model("large")

def extract_embeddings(audio_file_path):
    # Load the audio file
    audio = whisper.load_audio(audio_file_path)
    
    # Pad/trim the audio to fit the model's expected input length
    audio = whisper.pad_or_trim(audio)
    
    # Make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    # Forward pass through the model to extract the encoder's features (embeddings)
    with torch.no_grad():
        # Encoder outputs a tuple (logits, hidden states, ...)
        hidden_states = model.encoder(mel)
        
    # The last hidden state is usually used as the embedding representation
    # We can take the last layer's output or average over time steps for pooling
    embeddings = hidden_states[0]  # shape: [1, time_steps, feature_dim]
    
    # Pooling over the time axis to reduce it to a fixed-size embedding
    # You can use mean pooling or max pooling; here, we'll use mean pooling
    embeddings = embeddings.mean(dim=1)  # Shape becomes [1, feature_dim]
    
    return embeddings.detach().cpu().numpy()  # Convert to numpy for downstream use

# Example usage
audio_path = 'your_audio_file.wav'
embeddings = extract_embeddings(audio_path)

print(f"Extracted embeddings shape: {embeddings.shape}")
```

### Explanation:

- **Model**: The `whisper.load_model("large")` loads the largest Whisper model, which provides high-quality embeddings. You can choose between `base`, `small`, `medium`, or `large` depending on your needs and available resources.
  
- **Audio Processing**: The audio is loaded and converted to a **log-mel spectrogram**. This spectrogram is then fed into Whisper’s encoder to extract the embeddings.

- **Pooling**: We apply **mean pooling** across the time steps of the spectrogram to generate a fixed-size feature vector representing the audio.

- **Embeddings**: The resulting embeddings are returned as a NumPy array, which you can then use for classification, clustering, or visualization.

### Step 3: Use Embeddings for Downstream Tasks

Once you have the embeddings, you can use them for various speech-related tasks. For example, you can use machine learning models like SVM, k-NN, or neural networks to classify the speech or cluster audio samples based on their embeddings.

### Example Use Case: Classifying Speech

After extracting embeddings, you can train a classifier like Support Vector Machine (SVM) on them:

```python
from sklearn.svm import SVC

# Example of using extracted embeddings with an SVM classifier
embeddings_train = [...]  # List of embeddings from your training data
labels_train = [...]  # Corresponding labels for the training data

classifier = SVC()
classifier.fit(embeddings_train, labels_train)

# Example prediction using the extracted embeddings
predictions = classifier.predict([embeddings])
print(predictions)
```

## License

This repository is open-source and available under the MIT License.

---

### Notes:

- **Performance**: Whisper's larger models (`medium`, `large`) provide better feature representations but require more computational resources (GPU recommended). If you have limited resources, you can use smaller models, but the quality of embeddings may vary.
  
- **Audio Preprocessing**: Whisper works best with 16kHz mono-channel audio. You may want to preprocess your audio files accordingly (e.g., resampling using `librosa` or other tools).
  
- **Embedding Extraction and clasification**: Take a look at [Speech clasifation.md](https://github.com/sugarcane-mk/whisper/blob/main/Speech_calssification.md) for detailed procedure for data preparation, embedding extraction and classification.

---

### How to Run the Code

1. Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/whisper-embeddings.git
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Run the `extract_embeddings.py` script to generate embeddings for your audio file:

```bash
python extract_embeddings.py
```
3. Run the `classification_svm.py` script to classify embedings:

```bash
python classification_svm.py
```
---
Feel free to modify the code as needed for your specific task, and don’t hesitate to open issues or submit pull requests if you find any bugs or improvements!
Let me know if you need further modifications or explanations!
