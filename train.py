from torchvision import transforms
from omegaconf import DictConfig
import torch
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import EarlyStopping
import hydra
from dataset import CustomDataset
from encoders import timm_backbones
from pathlib import Path
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
import os 



@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def main(cfg: DictConfig) -> None:

    # args = parser.parse_args()
    hydra = HydraConfig.get()
    target_size = tuple(cfg.target_size)
    train_transform = transforms.Compose([
        transforms.Resize(target_size),  # Resize images to 224x224 pixels
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(target_size),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_dataset = CustomDataset(root=cfg.dataset.root_path, subset="train", transform=train_transform)
    val_dataset = CustomDataset(root=cfg.dataset.root_path, subset="val", transform=val_transform)
    test_dataset = CustomDataset(root=cfg.dataset.root_path, subset="test", transform=val_transform)
    

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    test_sampler = SequentialSampler(test_dataset)

    train_loader = DataLoader(train_dataset,sampler= train_sampler, batch_size= cfg.batch_size, num_workers=7)
    val_loader = DataLoader(val_dataset, sampler= val_sampler  ,batch_size= cfg.batch_size, num_workers=7)
    test_loader = DataLoader(test_dataset,sampler= test_sampler, batch_size= cfg.batch_size, num_workers=7)

    model = timm_backbones(encoder= cfg.model.encoder, num_classes=cfg.num_classes, optimizer_cfg=cfg.model.optimizer, l1_lambda=cfg.model.l1_lambda)
    # Define a checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath=hydra.runtime.output_dir +'/checkpoints/',  # Directory where checkpoints will be saved

        
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=5,
        verbose=False,
        mode='max'
    )

    # Define a TensorBoard logger to log metrics
    logger = TensorBoardLogger('logs', name='outputloggs')

    # Initialize a PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        min_epochs=cfg.min_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=logger, 
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback,early_stop_callback],
    )
    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Evaluate the model
    trainer.test(model, test_loader)
    state_dict = model.state_dict()
    model_name = hydra.runtime.output_dir +'/model.pth'
    torch.save(state_dict, model_name)


if __name__ == "__main__":
    main()