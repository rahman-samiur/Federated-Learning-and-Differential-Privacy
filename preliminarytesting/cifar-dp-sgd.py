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

# --- Opacus Imports ---
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
# --- End Opacus Imports ---

import warnings
# Ignore the specific UserWarning from PyTorch about TF32
warnings.filterwarnings('ignore', category=UserWarning,
                        module='torch.__init__')

# Set precision for better performance on modern GPUs
torch.set_float32_matmul_precision('medium')

# 1. The Model (Modified for Opacus)
# We must replace MaxPool2d with AvgPool2d and .view with nn.Flatten


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # --- MODIFIED ---
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        # --- END MOD ---

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # --- MODIFIED ---
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
        # --- END MOD ---

        self.flat_size = 64 * 8 * 8

        # --- MODIFIED ---
        self.flatten = nn.Flatten()
        # --- END MOD ---

        self.fc1 = nn.Linear(self.flat_size, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # (B, 3, 32, 32)
        x = self.pool1(self.relu1(self.conv1(x)))  # (B, 32, 16, 16)
        x = self.pool2(self.relu2(self.conv2(x)))  # (B, 64, 8, 8)
        # --- MODIFIED ---
        x = self.flatten(x)                          # (B, 64*8*8)
        # --- END MOD ---
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. The DataModule (Modified for Opacus)


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])

    def prepare_data(self):
        torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            full_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=True, transform=self.transform
            )
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
            num_workers=os.cpu_count(),
            # --- MODIFIED ---
            # drop_last=True is important for Opacus
            drop_last=True
            # --- END MOD ---
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

# 3. The LightningModule (Modified for Opacus)


class LitCIFAR10(pl.LightningModule):
    def __init__(self,
                 train_loader,  # We must pass the train_loader
                 epochs: int,
                 learning_rate: float,
                 target_epsilon: float,
                 target_delta: float,
                 max_grad_norm: float):

        super().__init__()
        # Save hyperparameters (Opacus params, LR, etc.)
        # We ignore 'train_loader' as it's not a simple hparam.
        self.save_hyperparameters(ignore=['train_loader'])

        # Store the loader and epochs for the PrivacyEngine
        self.train_loader = train_loader

        # --- MODIFIED ---
        # 1. Instantiate the model
        model = SimpleCNN(num_classes=10)
        # 2. Fix it for Opacus (replaces incompatible layers)
        self.model = ModuleValidator.fix(model)
        # --- END MOD ---

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._common_step(batch, batch_idx)
        self.train_acc(preds, y)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=True,
                 on_epoch=True, prog_bar=True)
        return loss

    # --- NEW METHOD ---
    # Log privacy budget (epsilon) at the end of each epoch
    def on_train_epoch_end(self):
        epsilon = self.privacy_engine.get_epsilon(self.hparams.target_delta)
        self.log('epsilon', epsilon, on_epoch=True, prog_bar=True)
    # --- END NEW ---

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._common_step(batch, batch_idx)
        self.val_acc(preds, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._common_step(batch, batch_idx)
        self.test_acc(preds, y)
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True)
        return loss

    # --- MODIFIED ---
    # This is where the magic happens
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate)

        # 1. Instantiate the PrivacyEngine
        self.privacy_engine = PrivacyEngine()

        # 2. Attach the PrivacyEngine to the model, optimizer, and data_loader
        # This will wrap them to perform DP-SGD
        self.model, optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            epochs=self.hparams.epochs,
            target_epsilon=self.hparams.target_epsilon,
            target_delta=self.hparams.target_delta,
            max_grad_norm=self.hparams.max_grad_norm,
        )

        return optimizer
    # --- END MOD ---

    # --- NEW METHOD ---
    # We must override train_dataloader to return the DP-wrapped loader
    def train_dataloader(self):
        return self.train_loader
    # --- END NEW ---


# 4. The Runner Script (Modified for Opacus)
if __name__ == "__main__":
    # --- Configuration ---
    BATCH_SIZE = 128
    MAX_EPOCHS = 10
    LEARNING_RATE = 1e-3

    # --- DP Configuration ---
    TARGET_EPSILON = 5.0
    TARGET_DELTA = 1e-5  # Usually 1 / len(dataset)
    MAX_GRAD_NORM = 1.2

    # --- Setup Data ---
    print("Setting up DataModule...")
    dm = CIFAR10DataModule(batch_size=BATCH_SIZE)
    dm.prepare_data()
    dm.setup('fit')  # Manually call setup to prepare self.train_dataset

    # --- MODIFIED ---
    # We must create the train_loader *before* the LightningModule
    print("Creating manual train_loader...")
    train_loader = dm.train_dataloader()

    # We set target_delta = 1 / len(train_dataset)
    DATASET_LEN = len(dm.train_dataset)
    TARGET_DELTA = 1.0 / DATASET_LEN
    print(f"Target Delta set to 1 / {DATASET_LEN} = {TARGET_DELTA}")

    # --- Setup Model ---
    print("Setting up LightningModule with Opacus parameters...")
    model = LitCIFAR10(
        train_loader=train_loader,
        epochs=MAX_EPOCHS,
        learning_rate=LEARNING_RATE,
        target_epsilon=TARGET_EPSILON,
        target_delta=TARGET_DELTA,
        max_grad_norm=MAX_GRAD_NORM
    )
    # --- END MOD ---
    logger = pl.loggers.TensorBoardLogger(
        "lightning_logs/", name="cifar_cnn_dp_sgd")

    print("Setting up Trainer...")
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        logger=logger,
        enable_checkpointing=False,
        log_every_n_steps=1  # Checkpointing DP models can be complex
    )

    # --- Training ---
    print("Starting DP training...")
    # Pass the datamodule for validation and testing
    trainer.fit(model, datamodule=dm)
        # --- Detach PrivacyEngine after training ---
    print("Detaching PrivacyEngine for testing...")
    model.privacy_engine.detach()
    # --- Testing ---
    print("Starting testing...")
    # trainer.test(model, datamodule=dm) # Use this if you pass 'model'
    trainer.test(datamodule=dm)  # Use this if 'model' was the one 'fit'
