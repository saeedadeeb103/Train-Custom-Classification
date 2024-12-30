import os
import zipfile
import requests
from tqdm import tqdm
from typing import List, Tuple
import numpy as np
from torch.utils.data import Dataset
import librosa
import torch

SAMPLE_RATE = 22050
DURATION = 1.4  # second

class EmodbDataset(Dataset):
    __url__ = "http://www.emodb.bilderbar.info/download/download.zip"
    __labels__ = ("angry", "happy", "neutral", "sad")
    __suffixes__ = {
        "angry": ["Wa", "Wb", "Wc", "Wd"],
        "happy": ["Fa", "Fb", "Fc", "Fd"],
        "neutral": ["Na", "Nb", "Nc", "Nd"],
        "sad": ["Ta", "Tb", "Tc", "Td"]
    }

    def __init__(self, root_path: str = './data/emodb', transform=None):
        super().__init__()
        self.root_path = root_path
        self.audio_root_path = os.path.join(root_path, "wav")
        
        # Ensure the dataset is downloaded
        self._ensure_dataset()

        ids = []
        targets = []
        for audio_file in os.listdir(self.audio_root_path):
            f_name, ext = os.path.splitext(audio_file)
            if ext != ".wav":
                continue

            suffix = f_name[-2:]
            for label, suffixes in self.__suffixes__.items():
                if suffix in suffixes:
                    ids.append(os.path.join(self.audio_root_path, audio_file))
                    targets.append(self.label2id(label))
                    break

        self.ids = ids
        self.targets = np.array(targets, dtype=np.int64)
        self.transform = transform

    def _ensure_dataset(self):
        """
        Ensures the dataset is downloaded and extracted.
        """
        if not os.path.isdir(self.audio_root_path):
            print(f"Dataset not found at {self.audio_root_path}. Downloading...")
            self._download_and_extract()

    def _download_and_extract(self):
        """
        Downloads and extracts the dataset zip file.
        """
        # Ensure the root path exists
        os.makedirs(self.root_path, exist_ok=True)

        # Download the dataset
        zip_path = os.path.join(self.root_path, "emodb.zip")
        with requests.get(self.__url__, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            with open(zip_path, "wb") as f, tqdm(
                desc="Downloading EMO-DB dataset",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))

        # Extract the dataset
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.root_path)

        # Clean up the zip file
        os.remove(zip_path)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple:
        target = self.targets[idx]
        audio = self.load_audio(self.ids[idx])  # Should return a numpy array

        if self.transform:
            print(f"Audio shape before transform: {audio.shape}")  # Debug
            audio = self.transform(audio)  # Apply transform
            print(f"Audio shape after transform: {audio.shape}")  # Debug

        return audio, target

    @staticmethod
    def id2label(idx: int) -> str:
        return EmodbDataset.__labels__[idx]

    @staticmethod
    def label2id(label: str) -> int:
        if label not in EmodbDataset.__labels__:
            raise ValueError(f"Unknown label: {label}")
        return EmodbDataset.__labels__.index(label)

    @staticmethod
    def load_audio(audio_file_path: str) -> np.ndarray:
        audio, sr = librosa.load(audio_file_path, sr=SAMPLE_RATE, duration=DURATION)
        assert SAMPLE_RATE == sr, "broken audio file"
        # Convert numpy array to PyTorch tensor
        return torch.tensor(audio, dtype=torch.float32)

    @staticmethod
    def get_labels() -> List[str]:
        return list(EmodbDataset.__labels__)
