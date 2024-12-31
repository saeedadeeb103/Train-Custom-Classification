# from datasets import get_dataset_by_name
# import numpy as np

# dataset = get_dataset_by_name("TESSDataset", root_path="./data/TESS", transform=None)

# print("Number of samples:", len(dataset))
# num_classes = len(np.unique(dataset.labels))
# print(f"Number of classes: {num_classes}")

# for idx in range(20):
#     embeddings, label = dataset[idx]
#     print(f"Sample {idx}: Embedding shape = {embeddings.shape}, Label = {label}")

from torch.multiprocessing import set_start_method
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Custom collate function to handle variable-length raw waveform inputs.
    Args:
        batch: List of tuples (tensor, label), where tensor has shape [sequence_length].
    Returns:
        padded_waveforms: Padded tensor of shape [batch_size, max_seq_len].
        attention_mask: Attention mask for padded sequences.
        labels: Tensor of shape [batch_size].
    """
    # Separate waveforms and labels
    waveforms, labels = zip(*batch)

    # Ensure waveforms are 1D tensors
    waveforms = [torch.tensor(waveform).squeeze() for waveform in waveforms]

    # Pad sequences to the same length
    padded_waveforms = pad_sequence(waveforms, batch_first=True)  # [batch_size, max_seq_len]
    

    # Create attention mask
    attention_mask = (padded_waveforms != 0).long()  # Mask for non-padded values
    # In the training loop or DataLoader debug


    # Convert labels to a tensor
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_waveforms, attention_mask, labels

if __name__ == "__main__":
    set_start_method("spawn")  # Ensure compatibility with Windows multiprocessing

    from datasets import get_dataset_by_name
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from encoders.transformer import Wav2Vec2Classifier
    from torch.utils.data import DataLoader
    import torch

    root_path = "./data/TESS"
    batch_size = 32
    num_classes = 7
    learning_rate = 5e-4

    dataset = get_dataset_by_name("TESSDataset", root_path=root_path, transform=None)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )

    val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )

    model = Wav2Vec2Classifier(num_classes=num_classes, learning_rate=learning_rate)

    logger = TensorBoardLogger("logs", name="TESS_Wav2Vec2")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    trainer = Trainer(
        max_epochs=20,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        num_sanity_val_steps=0,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, val_loader)
