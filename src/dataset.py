
from lightning.pytorch import LightningDataModule

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import EMNIST


class EMNISTDataModule(LightningDataModule):
    """Data module for the EMNIST dataset.

    Parameters
    ----------
    data_dir : str
        Directory to store the dataset.
    batch_size : int
        Number of samples per batch.
    image_size : int
        Size of the images.
    val_split : float
        Fraction of the training dataset to be used as validation.
    num_workers : int
        Number of workers to use for data loading.
    """

    def __init__(self, data_dir: str = './', batch_size: int = 64, val_split: float = 0.2,
                 image_size: int = 28, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=10),
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

    def prepare_data(self):
        """Download the EMNIST dataset.
        """
        EMNIST(self.data_dir, split='balanced', train=True, download=True)
        EMNIST(self.data_dir, split='balanced', train=False, download=True)

    def setup(self, stage=None):
        """Split the dataset into training, validation, and test sets.

        Parameters
        ----------
        stage : str, optional
            Stage in the data processing pipeline, by default None

        Raises
        ------
        ValueError
            If the stage is not one of 'fit', 'validate', or 'test'.
        """
        # Load the full EMNIST dataset for training
        full_train_dataset = EMNIST(
            self.data_dir, split='balanced', train=True, transform=self.transform)
        temp_tr_iterator = DataLoader(full_train_dataset, shuffle=False)
        # Calculate the sizes of each dataset split
        train_size = int((1 - self.val_split) * len(temp_tr_iterator))
        val_size = len(temp_tr_iterator) - train_size

        # Split the full dataset into training and validation datasets
        self.emnist_train, self.emnist_val = random_split(
            full_train_dataset, [train_size, val_size])

        # Load the EMNIST test dataset
        self.emnist_test = EMNIST(
            self.data_dir, split='balanced', train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.emnist_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.emnist_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.emnist_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
