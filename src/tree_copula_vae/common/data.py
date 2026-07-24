from abc import abstractmethod, ABCMeta
import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader, Dataset


class LightningDataModuleWrapper(pl.LightningDataModule, metaclass=ABCMeta):
    @property
    @abstractmethod
    def train_dataset(self) -> Dataset:
        pass

    @property
    @abstractmethod
    def val_dataset(self) -> Dataset:
        pass

    @property
    @abstractmethod
    def test_dataset(self) -> Dataset:
        pass

    @property
    @abstractmethod
    def predict_dataset(self) -> Dataset:
        pass

    def get_random_samples(self, train: bool = True, n_samples: int = 1):
        if train:
            ds = self.train_dataset
        else:
            ds = self.test_dataset


