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

class MSPPodcastDataset(Dataset):
    def __init__(self, labels_path: str, audio_dir: str, partition_path: str = None, transform=None):
        super().__init__()
        self.audio_dir = audio_dir
        self.labels_path = labels_path
        self.partition_path = partition_path
        self.transform = transform
        self.audio_files = []
        self.labels = []
        
        # Load labels and emotions
        self.labels_df = pd.read_csv(labels_path)
        self.emotions = sorted(self.labels_df['EmoClass'].unique().tolist())
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotions)}
        
        # Load partition information if provided
        self.partition_dict = self.load_partitions() if partition_path else {}
        
        # Ensure dataset exists
        self.download_dataset_if_not_exists()
        
        # Populate audio files and labels
        self.populate_dataset()

    def populate_dataset(self):
        """Populate audio file paths and corresponding labels."""
        for _, row in self.labels_df.iterrows():
            filename = row['FileName']
            emotion = row['EmoClass']
            
            # Check if file belongs to the partition (if partition_dict is provided)
            if self.partition_dict and self.partition_dict.get(filename, '') not in ['train', 'val', 'test']:
                continue
                
            audio_path = os.path.join(self.audio_dir, filename)
            if os.path.exists(audio_path) and emotion in self.emotion_to_idx:
                self.audio_files.append(audio_path)
                self.labels.append(self.emotion_to_idx[emotion])
        
        self.labels = torch.tensor(self.labels, dtype=torch.int64)

    def load_partitions(self) -> dict:
        """Load partition information from partitions.txt."""
        partition_dict = {}
        if not os.path.exists(self.partition_path):
            print(f"Partition file not found at {self.partition_path}. Treating all data as unpartitioned.")
            return partition_dict
        
        with open(self.partition_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                part, fname = line.split('; ')
                partition_dict[fname] = part.lower()
        
        return partition_dict

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        waveform = self.load_audio(audio_path)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label

    @staticmethod
    def load_audio(audio_path: str) -> torch.Tensor:
        """Load audio and ensure it's at the correct sample rate."""
        waveform, sr = torchaudio.load(audio_path)
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # Trim or pad to fixed duration
        target_length = int(SAMPLE_RATE * DURATION)
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        elif waveform.shape[1] < target_length:
            padding = torch.zeros((waveform.shape[0], target_length - waveform.shape[1]))
            waveform = torch.cat([waveform, padding], dim=1)
        
        # Normalize waveform
        waveform = normalize_waveform(waveform)
        return waveform

    def get_emotions(self) -> List[str]:
        return self.emotions

    def download_dataset_if_not_exists(self):
        """Placeholder for downloading MSP-Podcast dataset."""
        if not os.path.exists(self.audio_dir) or not os.path.exists(self.labels_path):
            print(f"Dataset not found at {self.audio_dir} or {self.labels_path}.")
            print("MSP-Podcast dataset requires manual download due to access restrictions.")
            print("Please download the dataset from the official source and place it in the specified paths.")
            raise FileNotFoundError("MSP-Podcast dataset not found.")
        print(f"Dataset found at {self.audio_dir} and {self.labels_path}.")
