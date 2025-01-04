import os
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
from typing import List, Tuple
import shutil
import kagglehub
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import subprocess
import zipfile
import os
# Constants (you may need to define these according to your requirements)
SAMPLE_RATE = 16000  # Define the sample rate for audio processing
DURATION = 3.0  # Duration of the audio in seconds

# Placeholder for waveform normalization
def normalize_waveform(audio: np.ndarray) -> torch.Tensor:
    # Convert to tensor if necessary
    if not isinstance(audio, torch.Tensor):
        audio = torch.tensor(audio, dtype=torch.float32)
    return (audio - torch.mean(audio)) / torch.std(audio)

class TESSRawWaveformDataset(Dataset):
    def __init__(self, root_path: str, transform=None):
        super().__init__()
        self.root_path = root_path
        self.audio_files = []
        self.labels = []
        self.emotions = ["happy", "sad", "angry", "neutral", "fear", "disgust", "surprise"]
        emotion_mapping = {e.lower(): idx for idx, e in enumerate(self.emotions)}
        self.download_dataset_if_not_exists()
        # Load file paths and labels from nested directories
        for root, dirs, files in os.walk(root_path):
            for file_name in files:
                if file_name.endswith(".wav"):
                    emotion_name = next(
                        (e for e in emotion_mapping if e in root.lower()), None
                    )
                    if emotion_name is not None:
                        self.audio_files.append(os.path.join(root, file_name))
                        self.labels.append(emotion_mapping[emotion_name])

        self.labels = np.array(self.labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load raw waveform and label
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        waveform = self.load_audio(audio_path)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label

    @staticmethod
    def load_audio(audio_path: str) -> torch.Tensor:
        # Load audio and ensure it's at the correct sample rate
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        assert sr == SAMPLE_RATE, f"Sample rate mismatch: expected {SAMPLE_RATE}, got {sr}"
        return normalize_waveform(audio)

    def get_emotions(self) -> List[str]:
        return self.emotions

    def download_dataset_if_not_exists(self):
      if not os.path.exists(self.root_path):
          print(f"Dataset not found at {self.root_path}. Downloading...")

          # Ensure the destination directory exists
          os.makedirs(self.root_path, exist_ok=True)

          # Download dataset using curl
          dataset_zip_path = os.path.join(self.root_path, "toronto-emotional-speech-set-tess.zip")
          curl_command = [
              "curl",
              "-L",
              "-o",
              dataset_zip_path,
              "https://www.kaggle.com/api/v1/datasets/download/ejlok1/toronto-emotional-speech-set-tess",
          ]

          try:
              subprocess.run(curl_command, check=True)
              print(f"Dataset downloaded to {dataset_zip_path}.")

              # Extract the downloaded zip file
              with zipfile.ZipFile(dataset_zip_path, "r") as zip_ref:
                  zip_ref.extractall(self.root_path)
              print(f"Dataset extracted to {self.root_path}.")

              # Remove the zip file to save space
              os.remove(dataset_zip_path)
              print(f"Removed zip file: {dataset_zip_path}")

          except subprocess.CalledProcessError as e:
              print(f"Error occurred during dataset download: {e}")
              raise


# Example usage
# dataset = TESSRawWaveformDataset(root_path="./TESS", transform=None)
# print("Number of samples:", len(dataset))