import ssl
import certifi

import os
import urllib.request
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms
from abc import ABCMeta, abstractmethod
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader, random_split
from tree_copula_vae.common.data import LightningDataModuleWrapper



try:
    _ctx = ssl.create_default_context(cafile=certifi.where())
    ssl._create_default_https_context = lambda *a, **k: _ctx
except Exception as e:
    # Fallback: If certifi is not installed or fails, simply disable verification (less secure but works for data download)
    print(f"Warning: Could not use certifi ({e}), disabling SSL verification completely.")
    ssl._create_default_https_context = ssl._create_unverified_context


# ---------------------------------------------------------
# 1. The Dataset Class (Logic of the screw)
# ---------------------------------------------------------
class ScrewDSpritesDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, download=True, tolerance=0.15):
        """
        Custom Dataset for 'The Screw' manifold.
        Args:
            split: Not used for file loading (since dSprites is one file),
                   but kept for compatibility signature.
        """
        self.transform = transform
        self.file_path = os.path.join(root_dir, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")

        # 1. Download
        if download and not os.path.exists(self.file_path):
            print(f"Downloading dSprites to {root_dir}...")
            os.makedirs(root_dir, exist_ok=True)
            url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
            urllib.request.urlretrieve(url, self.file_path)

        # 2. Load & Filter (The Screw Logic)
        if os.path.exists(self.file_path):  # Check to allow initialization before download in prepare_data
            data = np.load(self.file_path, encoding='latin1', allow_pickle=True)
            imgs = data['imgs']
            latents_values = data['latents_values']

            # Normalize Latents
            scale = latents_values[:, 2]
            orient = latents_values[:, 3]
            posX = latents_values[:, 4]

            scaler = MinMaxScaler()
            n_scale = scaler.fit_transform(scale.reshape(-1, 1)).flatten()
            n_orient = scaler.fit_transform(orient.reshape(-1, 1)).flatten()
            n_posX = scaler.fit_transform(posX.reshape(-1, 1)).flatten()

            # --- Manifold Rules ---
            # Rule 1: Scale ~ Orientation
            mask_1 = np.abs(n_scale - n_orient) < tolerance
            # Rule 2: Orientation ~ Inverse PosX
            mask_2 = np.abs(n_orient - (1.0 - n_posX)) < tolerance

            final_mask = mask_1 & mask_2

            # Keep only filtered data
            # dSprites is (N, 64, 64). We keep it as numpy for transforms to work if needed,
            # or convert to float tensor immediately.
            self.imgs = imgs[final_mask]
            self.latents = latents_values[final_mask]
        else:
            self.imgs = []
            self.latents = []

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Image is (64, 64)
        img = self.imgs[idx]

        # Convert to Tensor explicitly if no transform provided, ensuring (1, 64, 64)
        if self.transform:
            # Transforms usually expect (H, W, C) or PIL.
            # dSprites is (H, W). Let's make it compatible.
            img_tensor = torch.from_numpy(img).float().unsqueeze(0)  # (1, 64, 64)
            # Apply transform (Note: ensure your transforms handle 1-channel tensors)
            img = self.transform(img_tensor)
        else:
            img = torch.from_numpy(img).float().unsqueeze(0)

            #######################################
            # img = img * 0.9 + 0.05
            #######################################

        # We return latents too for validation/plotting
        latent = torch.from_numpy(self.latents[idx]).float()

        return img, latent

# ---------------------------------------------------------
# 2. The DataModule (Adapted to your structure)
# ---------------------------------------------------------
class ScrewDataModule(LightningDataModuleWrapper):
    """
    DataModule for the Synthetic 'Screw' Manifold (Correlated dSprites).
    """

    def __init__(
            self,
            data_dir: str,
            batch_size: int,
            test_batch_size: int = 1,
            pin_memory: bool = False,
            num_workers: int = 0,
            train_set_frac: float = 0.8,
            seed: int = 42,
            augment: bool = False,
            tolerance: float = 0.15
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.train_set_frac = train_set_frac
        self.seed = seed
        self.tolerance = tolerance

        # dSprites is black-and-white and synthetic. Regular augmentations (ColorJitter) are not relevant
        # and may damage the experiment. So transformations here are minimal.
        # Our Dataset already returns a normalized tensor, so we usually leave this empty.
        self.train_tfms = None
        self.eval_tfms = None

        self.screw_train = None
        self.screw_val = None
        self.screw_test = None
        self.screw_predict = None

    def prepare_data(self):
        # Trigger download
        ScrewDSpritesDataset(self.data_dir, download=True, tolerance=self.tolerance)

    def setup(self, stage: str = None):
        # Load the full dataset once
        full_dataset = ScrewDSpritesDataset(
            self.data_dir,
            transform=None,  # Transformations are handled internally or not needed for synthetic data
            download=False,
            tolerance=self.tolerance
        )

        # Calculate split sizes
        total_len = len(full_dataset)
        train_len = int(self.train_set_frac * total_len)
        val_len = int(0.1 * total_len)  # 10% Validation
        test_len = total_len - train_len - val_len

        # Deterministic Split
        generator = torch.Generator().manual_seed(self.seed)
        train_ds, val_ds, test_ds = random_split(
            full_dataset, [train_len, val_len, test_len], generator=generator
        )

        if stage == "fit" or stage is None:
            self.screw_train = train_ds
            self.screw_val = val_ds

            # Apply transforms if implemented (wrappers needed because random_split returns Subset)
            # For dSprites usually not needed, the data is already prepared as a tensor.

        if stage == "test" or stage is None:
            self.screw_test = test_ds

        if stage == "predict" or stage is None:
            self.screw_predict = test_ds

    def train_dataloader(self):
        return DataLoader(
            self.screw_train,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.screw_val,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(self.screw_test, batch_size=self.test_batch_size)

    def predict_dataloader(self):
        return DataLoader(self.screw_predict, batch_size=self.batch_size)

    # --- Implementation of Abstract Properties ---
    @property
    def train_dataset(self) -> Dataset:
        return self.screw_train

    @property
    def val_dataset(self) -> Dataset:
        return self.screw_val

    @property
    def test_dataset(self) -> Dataset:
        return self.screw_test

    @property
    def predict_dataset(self) -> Dataset:
        return self.screw_predict

