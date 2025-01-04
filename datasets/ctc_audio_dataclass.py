import os
import zipfile
import requests
from tqdm import tqdm
from typing import List, Tuple
import numpy as np
from torch.utils.data import Dataset
import librosa
import torch
from torch.nn.utils.rnn import pad_sequence

SAMPLE_RATE = 22050
DURATION = 1.4  # seconds

class CTCEmodbDataset(Dataset):
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
                    targets.append(self.label2id(label))  # Store as integers
                    break

        self.ids = ids
        self.targets = targets  # Target sequences as a list of lists
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
        os.makedirs(self.root_path, exist_ok=True)
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

        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.root_path)

        os.remove(zip_path)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Returns:
            x (torch.Tensor): Input sequence (audio features or waveform)
            y (torch.Tensor): Target sequence (labels or tokenized transcription)
            input_length (int): Length of input sequence
            target_length (int): Length of target sequence
        """
        target = torch.tensor([self.targets[idx]], dtype=torch.long) 
        audio = self.load_audio(self.ids[idx])  # Should return a numpy array

        if self.transform:
            audio = self.transform(audio)

        # Input length (for CTC)
        input_length = audio.shape[-1]  # Last dimension is the time dimension
        target_length = len(target)  # Length of target sequence

        return audio, target, input_length, target_length

    @staticmethod
    def id2label(idx: int) -> str:
        return CTCEmodbDataset.__labels__[idx]

    @staticmethod
    def label2id(label: str) -> int:
        if label not in CTCEmodbDataset.__labels__:
            raise ValueError(f"Unknown label: {label}")
        return CTCEmodbDataset.__labels__.index(label)

    @staticmethod
    def load_audio(audio_file_path: str) -> torch.Tensor:
        audio, sr = librosa.load(audio_file_path, sr=SAMPLE_RATE, duration=DURATION)
        assert SAMPLE_RATE == sr, "broken audio file"
        return torch.tensor(audio, dtype=torch.float32)

    @staticmethod
    def get_labels() -> List[str]:
        return list(CTCEmodbDataset.__labels__)
