
import torch
import numpy as np

from lightning.pytorch import LightningDataModule

from torch.utils.data import (
    DataLoader, random_split, Dataset)


class CustomDataset(Dataset):
    def __init__(
        self,
        generator, autoencoder,
        size: int, num_classes: int = 47,
        mode: str = "train", z_dim: int = 64,):
        if mode not in ["train", "val", "test"]:
            raise ValueError("mode must be in ['train', 'val', 'test']")
        
        self.generator = generator
        self.autoencoder = autoencoder
        self.size = size
        self.num_classes = num_classes
        self.mode = mode
        self.z_dim = z_dim
        self._setup_labels()
        self.generator.eval()
        self.autoencoder.eval()

    def _setup_labels(self):
        self.labels = torch.arange(self.num_classes).repeat(
            int(np.ceil(self.size / self.num_classes))).unsqueeze(1)
        if self.mode == "train":
            self.labels = self.labels[torch.randperm(
                self.labels.size(0))]

    def __getitem__(self, idx):
        noise = torch.randn(1, self.z_dim, 1, 1)
        with torch.no_grad():
            gen_img = self.generator(
                noise, self.labels[idx])
            gen_img = (gen_img - gen_img.min()) / (gen_img.max() - gen_img.min())
            mu, _ = self.autoencoder.encode(gen_img)
        return mu, self.labels[idx]
        
    def __len__(self):
        return self.size


class GenEMNISTDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 64, val_split: float = 0.2, 
                 num_workers: int = 4, num_classes: int = 47, 
                 train_size: int = 235000, test_size: int = 15000,
                 generator: torch.nn.Module = None, z_dim: int = 64,
                 autoencoder: torch.nn.Module = None,):
        super().__init__()
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.train_size = train_size
        self.test_size = test_size
        self.generator = generator
        self.autoencoder = autoencoder
        self.z_dim = z_dim
        
        self.train, self.val = random_split(
            CustomDataset(generator=self.generator, autoencoder=self.autoencoder, 
                          size=self.train_size, num_classes=self.num_classes, 
                          mode="train", z_dim=self.z_dim),
            [self.train_size - int(self.train_size * self.val_split),
             int(self.train_size * self.val_split)]
        )
        self.test = CustomDataset(generator=self.generator, autoencoder=self.autoencoder,
                                  size=self.test_size, num_classes=self.num_classes, 
                                  mode="test", z_dim=self.z_dim)
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
