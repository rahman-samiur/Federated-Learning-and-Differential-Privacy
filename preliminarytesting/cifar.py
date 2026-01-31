import warnings

# Ignore only UserWarnings
warnings.filterwarnings('ignore', category=UserWarning)
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy

# Set precision for better performance on modern GPUs
torch.set_float32_matmul_precision('medium')

# 1. The Model (Standard PyTorch)
# We define a simple CNN. This is a standard nn.Module.


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8

        # Calculate flattened size
        self.flat_size = 64 * 8 * 8

        self.fc1 = nn.Linear(self.flat_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # (B, 3, 32, 32)
        x = self.pool1(F.relu(self.conv1(x)))  # (B, 32, 16, 16)
        x = self.pool2(F.relu(self.conv2(x)))  # (B, 64, 8, 8)
        x = x.view(-1, self.flat_size)        # (B, 64*8*8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. The DataModule (PyTorch Lightning)
# This class handles all data-related tasks.


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # Normalize with mean and stddev of CIFAR-10
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])

    def prepare_data(self):
        # Download the dataset
        torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        # Called on every GPU
        if stage == "fit" or stage is None:
            # Load full training set
            full_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=True, transform=self.transform
            )
            # Split into 45k train, 5k validation
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [45000, 5000]
            )

        if stage == "test" or stage is None:
            self.test_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )

# 3. The LightningModule (PyTorch Lightning)
# This class ties the model, optimizer, and training logic together.


class LitCIFAR10(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        # Save hyperparameters (e.g., learning_rate)
        self.save_hyperparameters()

        self.model = SimpleCNN(num_classes=10)

        # Define loss function
        self.criterion = nn.CrossEntropyLoss()

        # Define metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        # Forward pass through the model
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        # Helper function for training and validation
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._common_step(batch, batch_idx)

        # Log metrics
        self.train_acc(preds, y)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=True,
                 on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._common_step(batch, batch_idx)

        # Log metrics
        self.val_acc(preds, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._common_step(batch, batch_idx)

        # Log metrics
        self.test_acc(preds, y)
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True)

        return loss

    def configure_optimizers(self):
        # We use the Adam optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate
        )
        return optimizer


# 4. The Runner Script
if __name__ == "__main__":
    # --- Configuration ---
    BATCH_SIZE = 128
    MAX_EPOCHS = 10
    LEARNING_RATE = 1e-3

    # --- Setup ---
    print("Setting up DataModule...")
    dm = CIFAR10DataModule(batch_size=BATCH_SIZE)

    print("Setting up LightningModule...")
    model = LitCIFAR10(learning_rate=LEARNING_RATE)
    logger = pl.loggers.TensorBoardLogger(
        "lightning_logs/", name="cifar_cnn")
    print("Setting up Trainer...")
    # The Trainer does all the heavy lifting
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",  # Automatically uses GPU if available
        devices="auto",
        logger=logger,         # Enables logging (e.g., for TensorBoard)
        enable_checkpointing=True,
        log_every_n_steps=1,
    )

    # --- Training ---
    print("Starting training...")
    trainer.fit(model, dm)

    # --- Testing ---
    print("Starting testing...")
    trainer.test(model, datamodule=dm)

    print("\n--- Done ---")
    print(f"View logs with: tensorboard --logdir=lightning_logs")

# End of cifar.py