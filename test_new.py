import torch
from torch.utils.data import DataLoader, SequentialSampler
from encoders.transformer import Wav2Vec2EmotionClassifier
from utils.helper_functions import collate_fn_transformer
from datasets import get_dataset_by_name
from omegaconf import OmegaConf
import librosa
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import os

# Preprocessing function for single audio input

def preprocess_audio(file_path, sample_rate=16000):
    waveform, sr = librosa.load(file_path, sr=sample_rate)
    waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return waveform

@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # Load the trained model
    hydra_cfg = HydraConfig.get()

    config = {
        "model_path": "model.pth",
        "dataset_name": "TESSDataset",
        "root_path": "data/TESS/TESS/YAF_angry",
        "batch_size": 32,
        "num_classes": 7,
        "sample_file": None,  # Provide a path to a single audio file for inference
        "test_folder": "data/test_TESS"
    }
    model = Wav2Vec2EmotionClassifier(num_classes=cfg.num_classes, optimizer_cfg=cfg.model.optimizer, learning_rate= cfg.model.optimizer.lr, freeze_base=True)
    model.load_state_dict(torch.load(config["model_path"]))
    model.eval()  # Set the model to evaluation mode

    sample_file = config["sample_file"]
    test_folder = config["test_folder"]
    if sample_file:
        # Test on a single audio file
        waveform = preprocess_audio(sample_file)
        with torch.no_grad():
            logits = model(waveform)
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()

        print(f"Predicted Class: {predicted_class}")
        print(f"Probabilities: {probabilities}")
    elif test_folder:
        import glob
        from sklearn.metrics import accuracy_score, classification_report

        audio_files = glob.glob(os.path.join(test_folder, "**/*.wav"), recursive=True)
        predictions = []
        actual_labels = []

        # Mapping folder names to labels based on emotions
        emotions = ["happy", "sad", "angry", "neutral", "fear", "disgust", "surprise"]
        label_mapping = {str(idx): emotion for idx, emotion in enumerate(emotions)}

        for file_path in audio_files:
            # Infer the true label from the parent folder name
            true_label_name = os.path.basename(os.path.dirname(file_path))
            true_label = int(true_label_name)  # Folder name is the numeric label

            waveform = preprocess_audio(file_path)
            with torch.no_grad():
                logits = model(waveform)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()

            # Store predictions and actual labels for evaluation
            predictions.append(predicted_class)
            actual_labels.append(true_label)

            # Print individual file prediction and match status
            match_status = "CORRECT" if predicted_class == true_label else "INCORRECT"
            print(f"{file_path} -> Predicted: {predicted_class} ({label_mapping[str(predicted_class)]}), "
                f"Actual: {true_label} ({label_mapping[str(true_label)]}) [{match_status}]")

        # Print overall accuracy and classification report
        accuracy = accuracy_score(actual_labels, predictions)
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(actual_labels, predictions, target_names=emotions))
    else:
        # Test on a dataset
        transform = None  # Use the appropriate transform if needed
        dataset = get_dataset_by_name(config["dataset_name"], root_path=config["root_path"], transform=transform)
        test_loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            sampler=SequentialSampler(dataset),
            collate_fn=collate_fn_transformer,
            num_workers=2  # Adjust based on your system
        )
        
        accuracy = 0
        total_samples = 0

        with torch.no_grad():
            for batch in test_loader:
                x, attention_mask, y = batch
                logits = model(x, attention_mask=attention_mask)
                predictions = torch.argmax(logits, dim=-1)

                accuracy += (predictions == y).sum().item()
                total_samples += y.size(0)

        print(f"Test Accuracy: {accuracy / total_samples:.4f}")

if __name__ == "__main__":
    # Example configuration (replace with actual config values)
    main()