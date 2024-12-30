from torchvision import transforms
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
import hydra
from datasets import list_datasets, get_dataset_by_name
from encoders import timm_backbones
from torchaudio import transforms as T
from hydra.core.hydra_config import HydraConfig
from utils.random_split import stratified_random_split, normalize_ratios
from utils.helper_functions import collate_fn


class AudioTransform:
    def __init__(self, sample_rate=22050, n_mels=128, n_fft=1024, hop_length=512):
        self.transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )

    def __call__(self, waveform: torch.Tensor):
        # Convert waveform to Mel-spectrogram
        mel_spec = self.transform(waveform)
        mel_spec_db = T.AmplitudeToDB()(mel_spec)

        # Add channel dimension (single channel)
        mel_spec_db = mel_spec_db.unsqueeze(0)  # Shape: [1, height, width]
        return mel_spec_db



@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()

    # Determine dataset name based on input type
    dataset_name = cfg.dataset_name

    # Print available datasets for debugging
    print(f"Available datasets: {list_datasets()}")
    print(f"Using dataset: {dataset_name}")

    # Define appropriate transformations
    if cfg.input_type == "image":
        target_size = tuple(cfg.dataset.target_size)
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = get_dataset_by_name(dataset_name, root_path=cfg.dataset.root_path, subset="train", transform=transform)
        val_dataset = get_dataset_by_name(dataset_name, root_path=cfg.dataset.root_path, subset="val", transform=transform)
        test_dataset = get_dataset_by_name(dataset_name, root_path=cfg.dataset.root_path, subset="test", transform=transform)
    elif cfg.input_type == "audio":
        transform = AudioTransform(sample_rate=cfg.dataset.sample_rate, n_mels=cfg.dataset.n_mels, n_fft= cfg.dataset.n_fft)
        dataset = get_dataset_by_name(dataset_name, root_path=cfg.dataset.root_path, transform=transform)
        normalized_ratios = normalize_ratios(cfg.dataset.split_ratios)
        splits = stratified_random_split(dataset, parts=normalized_ratios, targets=dataset.targets)
        train_dataset, val_dataset, test_dataset = splits
    else:
        raise ValueError(f"Unsupported input_type: {cfg.input_type}")
    
    # Define data loaders
    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=cfg.batch_size, num_workers=7, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=cfg.batch_size, num_workers=7, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=cfg.batch_size, num_workers=7, collate_fn=collate_fn)

    # Initialize the model
    if cfg.model.encoder == "ctc":
        # Use CTC encoder
        from encoders import CTCEncoderPL  # Import your CTC encoder class
        from models.CTCencoder import CTCEncoder  # Import the encoder backbone

        ctc_encoder = CTCEncoder(
            num_classes=cfg.num_classes,
            cnn_output_dim=256,  # Match the dimension of your Mel-spectrogram or features
            rnn_hidden_dim=256,
            rnn_layers=3
        )

        model = CTCEncoderPL(
            ctc_encoder=ctc_encoder,
            num_classes=cfg.num_classes,
            optimizer_cfg=cfg.model.optimizer
        )
    else:
        # Use the timm_backbones encoder
        from encoders import timm_backbones

        model = timm_backbones(
            encoder=cfg.model.encoder,
            num_classes=cfg.num_classes,
            optimizer_cfg=cfg.model.optimizer,
            l1_lambda=cfg.model.l1_lambda
        )
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath=f"{hydra_cfg.runtime.output_dir}/checkpoints/"
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=5,
        verbose=False,
        mode='min'
    )

    # Define logger
    logger = TensorBoardLogger(save_dir="logs", name="outputloggs")

    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        min_epochs=cfg.min_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=logger,
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Evaluate the model
    trainer.test(model, test_loader)

    # Save the trained model
    model_path = f"{hydra_cfg.runtime.output_dir}/model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()