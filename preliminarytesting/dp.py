import inspect
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torch.utils.data import DataLoader, random_split, Dataset  # Added Dataset
from torchvision import transforms

# --- NEW IMPORTS for Scikit-Learn ---
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
# --- END NEW IMPORTS ---


# Set a seed for reproducibility
pl.seed_everything(42)

# --- 1. The LightningModule (The Model) ---
#
#     This is your simple CNN, which is a good baseline
#     for your DP experiment.
#


class LFWClassifier(pl.LightningModule):
    """
    A simple CNN for LFW classification.
    """

    def __init__(self, num_classes, learning_rate=1e-3):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # --- CNN Architecture ---
        # Input images are 3 x 128 x 128
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # -> 32 x 128 x 128
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # -> 32 x 64 x 64
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # -> 64 x 64 x 64
        # -> after pool: 64 x 32 x 32
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # -> 128 x 32 x 32
        # -> after pool: 128 x 16 x 16

        # Flattened size: 128 * 16 * 16 = 32768
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

        # --- Metrics ---
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes)

    def forward(self, x):
        """Defines the forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the features
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def _common_step(self, batch, batch_idx):
        """Common logic for training, validation, and test steps."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch, batch_idx)
        # Log metrics
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch, batch_idx)
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch, batch_idx)
        # Log metrics
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', acc, on_epoch=True)

    def configure_optimizers(self):
        """Define the optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# --- NEW: Custom Dataset Wrapper ---
class SklearnLFWDataset(Dataset):
    """
    A custom torch Dataset to wrap the numpy arrays from scikit-learn.
    """

    def __init__(self, images, targets, transform=None):
        self.images = images  # This is a (N, H, W, C) numpy array
        self.targets = targets  # This is a (N,) numpy array
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image_np = self.images[idx]
        target_val = self.targets[idx]

        # scikit-learn gives (H, W, C) numpy array, values [0, 255]
        # transforms.Resize expects a PIL Image.
        image_pil = Image.fromarray(image_np.astype('uint8'), 'RGB')

        # Apply transforms (Resize, ToTensor, Normalize)
        if self.transform:
            image_tensor = self.transform(image_pil)

        # Convert target to a long tensor
        target = torch.tensor(target_val, dtype=torch.long)

        return image_tensor, target


# --- 2. The LightningDataModule (The Data) ---
#
#     *** UPDATED WITH DATA AUGMENTATION ***
#
class LFWDataModule(pl.LightningDataModule):
    """
    DataModule for the LFW (Labeled Faces in the Wild) dataset.
    
    This version uses scikit-learn's `fetch_lfw_people` to download
    and filter the dataset based on `min_faces_per_person`.
    
    It then manually splits the data into train, validation, and test sets.
    """

    def __init__(self, data_dir: str = "./lfw_data_sklearn",
                 batch_size: int = 32,
                 min_faces: int = 50,
                 resize_dim: tuple = (128, 128),
                 num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.min_faces = min_faces
        self.resize_dim = resize_dim
        self.num_workers = num_workers
        self.num_classes = 0  # Will be set in setup()

        # --- UPDATED: Define TWO sets of transforms ---

        # 1. Transform for TRAINING data (with augmentation)
        self.train_transform = transforms.Compose([
            transforms.Resize(self.resize_dim),
            transforms.RandomHorizontalFlip(),  # Randomly flip images
            # Randomly rotate by +/- 10 degrees
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # 2. Transform for VALIDATION and TEST data (no augmentation)
        # We never augment validation or test data!
        self.val_test_transform = transforms.Compose([
            transforms.Resize(self.resize_dim),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def prepare_data(self):
        """Downloads the data (if not present) using scikit-learn."""
        print("Downloading LFW data via scikit-learn...")
        fetch_lfw_people(
            data_home=self.data_dir,
            min_faces_per_person=self.min_faces,
            color=True,          # Get 3-channel images
            resize=1.0,          # Get 250x250, our transform will resize
            download_if_missing=True
        )
        print("Download complete.")

    def setup(self, stage: str = None):
        """
        Loads and splits the data from scikit-learn.
        This is called on every GPU in a distributed setting.
        """
        # We load/split once and cache the splits
        if not hasattr(self, 'X_train'):
            print("Loading and splitting LFW data...")
            lfw_people = fetch_lfw_people(
                data_home=self.data_dir,
                min_faces_per_person=self.min_faces,
                color=True,
                resize=1.0
            )

            X = lfw_people.images  # (n_samples, 250, 250, 3)
            y = lfw_people.target  # (n_samples,)
            self.num_classes = lfw_people.target_names.shape[0]

            # Split 1: 80% for Train+Val, 20% for Test
            # stratify=y ensures class distribution is similar in train/test
            X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

            # Split 2: 80% of Train+Val for Train, 20% for Val
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)

            print(f"Data loaded. Found {self.num_classes} classes.")

        # Assign datasets based on the stage
        if stage == 'fit' or stage is None:
            # --- UPDATED: Use the correct transform ---
            self.lfw_train = SklearnLFWDataset(
                self.X_train, self.y_train, transform=self.train_transform)
            self.lfw_val = SklearnLFWDataset(
                self.X_val, self.y_val, transform=self.val_test_transform)
            print(
                f"Fit stage: Train samples: {len(self.lfw_train)}, Val samples: {len(self.lfw_val)}")

        if stage == 'test' or stage is None:
            # --- UPDATED: Use the correct transform ---
            self.lfw_test = SklearnLFWDataset(
                self.X_test, self.y_test, transform=self.val_test_transform)
            print(f"Test stage: Test samples: {len(self.lfw_test)}")

    def train_dataloader(self):
        return DataLoader(self.lfw_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.lfw_val, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.lfw_test, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)


# --- 3. The Trainer (The Runner) ---

if __name__ == '__main__':

    # --- Configuration ---
    # Using a different data dir to avoid conflicts
    DATA_DIR = "./lfw_data_sklearn"
    BATCH_SIZE = 64
    # Setting min_faces=70 gives 7 classes in the sklearn version
    MIN_FACES_PER_PERSON = 70
    RESIZE_DIM = (128, 128)
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-3

    # --- Initialization ---

    # 1. Initialize DataModule
    datamodule = LFWDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        min_faces=MIN_FACES_PER_PERSON,
        resize_dim=RESIZE_DIM,
        num_workers=os.cpu_count() // 2
    )

    # We must call prepare_data() and setup() to access datamodule.num_classes
    print("Preparing data...")
    datamodule.prepare_data()
    # This will also set up 'test' data due to our logic
    datamodule.setup('fit')

    print(f"Data setup complete. Found {datamodule.num_classes} classes.")

    # 2. Initialize Model
    model = LFWClassifier(num_classes=datamodule.num_classes,
                          learning_rate=LEARNING_RATE)

    # 3. Initialize Trainer
    # Using TensorBoard for logging
    logger = pl.loggers.TensorBoardLogger(
        "lightning_logs/", name="lfw_cnn_sklearn_aug")

    # Add callbacks for early stopping and checkpointing
    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        pl.callbacks.ModelCheckpoint(
            monitor="val_acc", dirpath="checkpoints/", filename="best-lfw-model", mode="max")
    ]

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="auto",  # Automatically uses GPU/MPS if available
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10
    )

    # --- Training ---
    print("Starting training...")
    trainer.fit(model, datamodule)

    # --- Testing ---
    print("Training finished. Starting testing...")

    # Load the best model from checkpoint for testing
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"Loading best model from: {best_model_path}")

    # Check if best_model_path is valid
    if best_model_path and os.path.exists(best_model_path):
        best_model = LFWClassifier.load_from_checkpoint(best_model_path)
        # Run testing on the 'test' split
        trainer.test(best_model, datamodule=datamodule)
    else:
        print("No best model checkpoint found. Testing with the final model.")
        trainer.test(model, datamodule=datamodule)

    print("\n--- Done ---")
    print(f"View logs with: tensorboard --logdir=lightning_logs")
