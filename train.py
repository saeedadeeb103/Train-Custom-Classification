from torchvision import transforms
from omegaconf import DictConfig
import torch
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import EarlyStopping
import hydra
from dataset import CustomDataset
from encoders import timm_backbones
from pathlib import Path
from omegaconf import OmegaConf



@hydra.main(config_path="configs",config_name="train")
def main(cfg: DictConfig):
    target_size = tuple(cfg.model.target_size)
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
    train_dataset = CustomDataset(root=cfg.dataset.train_path, transform=train_transform)
    val_dataset = CustomDataset(root=cfg.dataset.val_path, transform=val_transform)
    test_dataset = CustomDataset(root=cfg.dataset.test_path, transform=val_transform)
    

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    test_sampler = SequentialSampler(test_dataset)

    train_loader = DataLoader(train_dataset,sampler= train_sampler, batch_size= cfg.model.batch_size)
    val_loader = DataLoader(val_dataset, sampler= val_sampler  ,batch_size= cfg.model.batch_size )
    test_loader = DataLoader(test_dataset,sampler= test_sampler, batch_size= cfg.model.batch_size)

    model = timm_backbones(encoder= cfg.model.encoder, num_classes=len(train_dataset.classes), optimizer_cfg=cfg.model.optimizer)
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
        max_epochs=cfg.model.max_epochs,
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
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument("--encoder", type=str, default="mobilenetv2_100", help="Encoder architecture")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=3e-4, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=20, help="Maximum number of epochs")
    parser.add_argument("--target_size", nargs="+", type=int, default=[224, 224], help="Target size for image resizing")
    parser.add_argument("--train_path", type=str, default="/home/saeed101/projects/Train-Custom-Classification/dataset/train", help="Path to the training dataset")
    parser.add_argument("--val_path", type=str, default="/home/saeed101/projects/Train-Custom-Classification/dataset/valid", help="Path to the validation dataset")
    parser.add_argument("--test_path", type=str, default="/home/saeed101/projects/Train-Custom-Classification/dataset/test", help="Path to the test dataset")


    args = parser.parse_args()

    # Update the configuration using argparse arguments\
    cfg_override= {'model': 
                        {'encoder': args.encoder,
                        'num_classes': args.num_classes,
                        'batch_size': args.batch_size,
                        'target_size': args.target_size,
                        'max_epochs': args.max_epochs,
                        'optimizer': 
                            {'name': args.optimizer,
                            'lr': args.lr,
                            'weight_decay': args.weight_decay,
                            }
                        },
                    'dataset': 
                        {'train_path': args.train_path,
                        'val_path': args.val_path,
                        'test_path': args.test_path, 
                        }
            }

    # Load the original configuration
    cfg = OmegaConf.load("configs/train.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.create(cfg_override))
    main(cfg)