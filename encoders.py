import torch.optim as optim
import pytorch_lightning as pl
import timm
from torchmetrics import Accuracy
import torch


class timm_backbones(pl.LightningModule):
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
    def __init__(self, encoder='resnet18', num_classes=2, optimizer_cfg=None):
        super().__init__()

        self.encoder = encoder
        self.model = timm.create_model(encoder, pretrained=True)
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        # Modify the final classification layer to match the number of classes
        in_features = self.model.fc.in_features if hasattr(self.model, 'fc') else self.model.classifier.in_features
        final_layer = torch.nn.Linear(in_features, num_classes)
        if hasattr(self.model, 'fc'):
            self.model.fc = final_layer
        else:
            self.model.classifier = final_layer

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
        return self.model(x)

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


