# ===================== Extract embeddings from specific layer ==================================================

from google.colab import drive
import os
import librosa
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperModel, Wav2Vec2Processor, Wav2Vec2Model
import torch.nn.functional as F

# Connect do google drive to access data
drive.mount('/content/drive')

# Embedding extraction

# =====================================  Path to Audio directory ===================================================
data_dir = '/content/drive/Shareddrives/Priya_speechlab/UA_speech'

# ======================================== Setup wav2vec Model =====================================================
def load_whisper_model(model_name):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperModel.from_pretrained(model_name)
    return processor, model

# ======================================== Set to wav2vec Model ====================================================
def load_wav2vec_model(model_name):
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    return processor, model

# ======================================= Extract embeddings for Whisper ============================================

def extract_embeddings_whisper(audio_file, layer_index=-1): # Added layer_index parameter
    try:
        speech_array, sampling_rate = librosa.load(audio_file, sr=16000)
        inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        mel_features = inputs['input_features']

        target_length = 3000
        current_length = mel_features.shape[2]

        if current_length < target_length:
            padding_length = target_length - current_length
            mel_features = F.pad(mel_features, (0, padding_length), value=0)
        elif current_length > target_length:
            mel_features = mel_features[:, :, :target_length]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        mel_features = mel_features.to(device)

        with torch.no_grad():
            outputs = model.encoder(input_features=mel_features, attention_mask=inputs.get('attention_mask'), output_hidden_states=True) # Get all hidden states

        # Extract embeddings from the specified layer
        embeddings = outputs.hidden_states[layer_index].mean(dim=1).squeeze().cpu().numpy()
        return embeddings

    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None


# ======================================= Extract embeddings for Wav2vec ============================================

def extract_embeddings_wav2vec(audio_file, layer_index=-1): # Added layer_index parameter
    try:
        speech_array, sampling_rate = librosa.load(audio_file, sr=16000)
        inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        input_values = inputs['input_values']

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_values = input_values.to(device)

        with torch.no_grad():
            outputs = model(input_values, attention_mask=inputs.get('attention_mask'), output_hidden_states=True) # Get all hidden states

        # Extract embeddings from the specified layer
        embeddings = outputs.hidden_states[layer_index].mean(dim=1).squeeze().cpu().numpy()
        return embeddings

    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None


# ======================================= Extraction call for Whisper ============================================
model_name = "openai/whisper-small"
processor, model = load_whisper_model(model_name)

model.eval()

# ======================================= Path to store embeddings ============================================

output_dir = '/home/ssn/priya/speech_embeddings/UAspeechB1/whisper_small'   # embedding path
speaker_folders = os.listdir(data_dir) # Create sub-folders in output dir
for speaker in speaker_folders:
    speaker_path = os.path.join(output_dir, speaker)
    os.makedirs(speaker_path, exist_ok=True)
for root, _, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".wav"):
            audio_path = os.path.join(root, file)
            relative_path = os.path.relpath(audio_path, data_dir)
            parts = relative_path.split(os.sep)

            # Construct the output filename
            if len(parts) >= 3:  # Assuming the structure is data/class_label/speaker/audio.wav
                speaker = parts[0]
                audio_name = os.path.splitext(file)[0]
                output_filename = os.path.join(output_dir, speaker, f"{audio_name}_embedding.npy")

                # Extract and save the embeddings
                embeddings = extract_embeddings_whisper(audio_path,7) # from 8th layer
                if embeddings is not None:
                    np.save(output_filename, embeddings)
                    print(f"Embeddings saved to {output_filename}")

print ("Embedding extraction successfully completed.")

# ======================================= Extraction call for Wav2vec ============================================

# Wav2vec extraction
model_name = "facebook/wav2vec2-base-960h"
processor, model = load_wav2vec_model(model_name)

output_dir = '/home/ssn/priya/speech_embeddings/UAspeechB1/wav2vec_base'  # embedding path
speaker_folders = os.listdir(data_dir) # Create sub-folders in output dir
for speaker in speaker_folders:
    speaker_path = os.path.join(output_dir, speaker)
    os.makedirs(speaker_path, exist_ok=True)

# Extraction call
for root, _, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".wav"):
            audio_path = os.path.join(root, file)
            relative_path = os.path.relpath(audio_path, data_dir)
            parts = relative_path.split(os.sep)

            if len(parts) >= 3:  # Assuming the structure is data/class_label/speaker/audio.wav
                speaker = parts[0]
                audio_name = os.path.splitext(file)[0]
                output_filename = os.path.join(output_dir, speaker, f"{audio_name}_embedding.npy")

                embeddings = extract_embeddings_wav2vec(audio_path, 8)
                if embeddings is not None:
                    np.save(output_filename, embeddings)
                    print(f"Embeddings saved to {output_filename}")

print ("Embedding extraction successfully completed.")
