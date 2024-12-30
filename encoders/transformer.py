import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score
from transformers import Wav2Vec2Model
import torch.nn.functional as F


class Wav2Vec2Classifier(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-4, l1_lambda=0.0):
        super(Wav2Vec2Classifier, self).__init__()
        self.save_hyperparameters()

        # Wav2Vec2 backbone
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

        # Classification head
        self.classifier = torch.nn.Linear(self.wav2vec2.config.hidden_size, num_classes)

        # Metrics
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass", num_classes=num_classes)
        self.recall = Recall(task="multiclass", num_classes=num_classes)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes)

        self.l1_lambda = l1_lambda

    def forward(self, x, attention_mask=None):
        # Debug input shape
        print(f"Original input shape: {x.shape}")

        # Ensure input shape is [batch_size, sequence_length]
        if x.dim() > 2:
            x = x.squeeze(-1)  # Remove unnecessary dimensions if present

        print(f"Shape after squeeze: {x.shape}")

        # Pass through Wav2Vec2 backbone
        output = self.wav2vec2(x, attention_mask=attention_mask)
        x = output.last_hidden_state

        # Classification head
        x = torch.mean(x, dim=1)  # Pooling
        logits = self.classifier(x)
        return logits


    def training_step(self, batch, batch_idx):
        x, attention_mask, y = batch
        print(f"Input shape: {x.shape}, Attention mask shape: {attention_mask.shape}")

        # Forward pass
        logits = self(x, attention_mask=attention_mask)

        # Compute loss
        loss = F.cross_entropy(logits, y)

        # Add L1 regularization if specified
        l1_norm = sum(param.abs().sum() for param in self.parameters())
        loss += self.l1_lambda * l1_norm

        # Log metrics
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, attention_mask, y = batch  # Unpack batch

        # Forward pass
        logits = self(x, attention_mask=attention_mask)


        # Compute loss and metrics
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        accuracy = self.accuracy(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        f1 = self.f1(preds, y)

        # Log metrics
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", accuracy, prog_bar=True, logger=True)
        self.log("val_precision", precision, prog_bar=True, logger=True)
        self.log("val_recall", recall, prog_bar=True, logger=True)
        self.log("val_f1", f1, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, attention_mask, y = batch  # Unpack batch

        # Forward pass
        logits = self(x, attention_mask=attention_mask)


        # Compute loss and metrics
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        accuracy = self.accuracy(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        f1 = self.f1(preds, y)

        # Log metrics
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_acc", accuracy, prog_bar=True, logger=True)
        self.log("test_precision", precision, prog_bar=True, logger=True)
        self.log("test_recall", recall, prog_bar=True, logger=True)
        self.log("test_f1", f1, prog_bar=True, logger=True)

        return {"test_loss": loss, "test_accuracy": accuracy}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=5, min_lr=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
