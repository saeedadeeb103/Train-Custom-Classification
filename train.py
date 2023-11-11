import os 
from PIL import Image 
from torchvision import transforms
from omegaconf import DictConfig
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
import timm
from pytorch_lightning.callbacks import EarlyStopping
import hydra
from torchmetrics import Accuracy
from dataset import CustomDataset

class ResNet18(pl.LightningModule):
    """
    PyTorch Lightning model for image classification using a ResNet-18 architecture.

    This model uses a pre-trained ResNet-18 model and fine-tunes it for a specific number of classes.

    Args:
        num_classes (int, optional): The number of classes in the dataset. Defaults to 2.
        optimizer_cfg (DictConfig, optional): A Hydra configuration object for the optimizer.

    Methods:
        forward(x): Computes the forward pass of the model.
        configure_optimizers(): Configures the optimizer for the model.
        training_step(batch, batch_idx): Performs a training step on the model.
        validation_step(batch, batch_idx): Performs a validation step on the model.
        on_validation_epoch_end(): Called at the end of each validation epoch.
        test_step(batch, batch_idx): Performs a test step on the model.

    Example:
        model = ResNet18(num_classes=2, optimizer_cfg=cfg.model.optimizer)
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        trainer.test(model, dataloaders=test_dataloader)
    """
    def __init__(self, num_classes=2, optimizer_cfg=None):
        super().__init__()

        # Load a pretrained ResNet-18 model from timm
        self.resnet = timm.create_model('resnet18', pretrained=True)
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        # Modify the final classification layer to match the number of classes in your dataset
        in_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(in_features, num_classes)
        if optimizer_cfg is not None:
            optimizer_name = optimizer_cfg.name
            optimizer_lr = optimizer_cfg.lr
            optimizer_weight_decay = optimizer_cfg.weight_decay

            if optimizer_name == 'Adam':
                self.optimizer = optim.Adam(self.parameters(), lr=optimizer_lr, weight_decay=optimizer_weight_decay)
            elif optimizer_name == 'SGD':
                # You can add more optimizers as needed
                self.optimizer = optim.SGD(self.parameters(), lr=optimizer_lr, weight_decay=optimizer_weight_decay)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        else:
            self.optimizer = None
        print("Optimizer Used:", self.optimizer)

    def forward(self, x):
        return self.resnet(x)

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()

        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)

        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()

        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        accuracy = self.accuracy(y, preds)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log('val_acc', accuracy, prog_bar=True, on_epoch=True, on_step=True)

        return loss

    def on_validation_epoch_end(self):
        avg_loss = self.trainer.logged_metrics['val_loss_epoch']
        accuracy = self.trainer.logged_metrics['val_acc_epoch']

        self.log('val_loss', avg_loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', accuracy, prog_bar=True, on_epoch=True)

        return {'Average Loss:': avg_loss, 'Accuracy:': accuracy}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        return {'test_loss': loss, 'test_preds': preds, 'test_targets': y}

@hydra.main(config_path="configs",config_name="train")
def main(cfg: DictConfig):
    optimizer = 'SGD'
    target_size = (224, 224)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(target_size),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomInvert(),
        transforms.RandomAutocontrast()
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(target_size),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_dataset = CustomDataset(root="/home/saeed101/projects/ml/dataset/train", transform=train_transform)
    val_dataset = CustomDataset(root="/home/saeed101/projects/ml/dataset/valid", transform=val_transform)
    test_dataset = CustomDataset(root="/home/saeed101/projects/ml/dataset/test", transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size= 4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size= 4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size= 4, shuffle=False)

    model = ResNet18(num_classes=len(train_dataset.classes), optimizer_cfg=cfg.model.optimizer)
    # Define a checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath='checkpoints/',  # Directory where checkpoints will be saved

        
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='max'
    )

    # Define a TensorBoard logger to log metrics
    # logger = TensorBoardLogger('logs', name='simple_cnn')

    # Initialize a PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        # logger=logger, 
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback,early_stop_callback],

        
    )
    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Evaluate the model
    trainer.test(model, test_loader)
    state_dict = model.state_dict()
    model_name = 'model.pth'
    torch.save(state_dict, model_name)


if __name__ == "__main__":
    main()