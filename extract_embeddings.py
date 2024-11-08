import os
import librosa
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperModel
import torch.nn.functional as F

# Define paths and parameters
data_dir = "/path/to/audio_dir"  # Replace with the actual path to your data directory
output_dir = "/path/to/save_emneddings"  # Replace with the desired output directory
model_name = "openai/whisper-small"  # You can choose other model variants like whisper-base, whisper-large, etc.

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load WhisperProcessor and WhisperModel
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperModel.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()

# Function to extract embeddings
def extract_embeddings(audio_file):
    try:
        # Load the audio file using librosa (Resample to 16000 Hz, which Whisper expects)
        speech_array, sampling_rate = librosa.load(audio_file, sr=16000)
        
        # Preprocess the audio using the processor (this will create mel-spectrogram features)
        inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

        # Get the mel-spectrogram features from the processor's output
        mel_features = inputs['input_features']

        # Print the shape of the mel-spectrogram to debug
        print(f"Mel-spectrogram shape: {mel_features.shape}")

        # Whisper expects mel-spectrograms of length 3000
        target_length = 3000  # Whisper model's expected length
        current_length = mel_features.shape[2]  # Access time frames dimension

        if current_length < target_length:
            # If the features are shorter than the target, pad with zeros
            padding_length = target_length - current_length
            mel_features = F.pad(mel_features, (0, padding_length), value=0)
        elif current_length > target_length:
            # If the features are longer than the target, truncate to 3000 time frames
            mel_features = mel_features[:, :, :target_length]

        # Move the inputs and model to the same device (GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        mel_features = mel_features.to(device)

        # Forward pass through the model to extract embeddings
        # We don't need the decoder for feature extraction, so we'll pass only the encoder inputs
        with torch.no_grad():
            # Pass the features only through the encoder part of the model (no decoder inputs required)
            outputs = model.encoder(input_features=mel_features, attention_mask=inputs.get('attention_mask'))

        # Extract the embeddings (mean across time steps)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # Averaging over time steps
        return embeddings

    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

# Iterate through all audio files in the data directory
for root, _, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".wav"):
            audio_path = os.path.join(root, file)
            audio_name = os.path.splitext(file)[0]  # Remove the file extension
            
            # Construct the output filename
            output_filename = os.path.join(output_dir, f"{audio_name}_embedding.npy")
            
            # Extract embeddings and save to file
            embeddings = extract_embeddings(audio_path)
            if embeddings is not None:
                np.save(output_filename, embeddings)
                print(f"Embeddings for {audio_name} saved to {output_filename}")
