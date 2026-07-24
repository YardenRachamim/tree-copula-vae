import torch
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from torchvision.datasets import FashionMNIST
from torchvision import transforms

from tree_copula_vae.common.data import LightningDataModuleWrapper
from tree_copula_vae.fashion_mnist.config import E_OBSERVED_DIST_TYPE


class AddNoiseToTensor(object):
    """
    Preprocessing as described in Appendix 4 (https://arxiv.org/src/1907.06845v4/anc/cont_bern_aux.pdf#page=6.09)
    Add custom transformation that adds uniform [0,1] noise
    to the integer pixel values between 0 and 255 and then
    divide by 256, to obtain values in [0,1]
    """

    def __call__(self, pic):
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        tensor = img.float()
        return (tensor + torch.rand(tensor.size())).div(256.)


class FashionMNISTDataModule(LightningDataModuleWrapper):
    def __init__(self,
                 data_dir: str,
                 batch_size: int,
                 pin_memory: bool = False,
                 num_workers: int = 0,
                 train_set_frac: float = .8,
                 seed: int = 42,
                 num_samples: int = None,
                 test_on_train: bool = False,
                 observed_distribution_type: E_OBSERVED_DIST_TYPE = E_OBSERVED_DIST_TYPE.Gaussian
                 ):
        super().__init__()
        self.data_dir = data_dir

        # --- Preprocessing identical to MNIST ---
        transformation = [
            AddNoiseToTensor(),
        ]

        if observed_distribution_type == E_OBSERVED_DIST_TYPE.Gaussian:
            transformation.append(transforms.Normalize((0.5,), (0.5,)))
        elif observed_distribution_type == E_OBSERVED_DIST_TYPE.Bernoulli:
            transformation.append(lambda x: x > 0.5)

        transformation.append(lambda x: x.type(torch.get_default_dtype()))

        self.transform = transforms.Compose(transformation)

        self.seed = seed
        self.train_set_frac = train_set_frac
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.fashion_train = None
        self.fashion_val = None
        self.fashion_test = None
        self.fashion_predict = None
        self.num_samples = num_samples
        self.test_on_train = test_on_train

    def prepare_data(self):
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            fashion_full = FashionMNIST(self.data_dir, train=True, transform=self.transform)
            if self.num_samples is not None:
                fashion_full = Subset(dataset=fashion_full, indices=torch.arange(self.num_samples))

            train_set_size = int(self.train_set_frac * len(fashion_full))
            valid_set_size = len(fashion_full) - train_set_size
            seed = torch.Generator().manual_seed(self.seed)

            if not self.test_on_train:
                self.fashion_train, self.fashion_val = random_split(fashion_full, [train_set_size, valid_set_size], generator=seed)
            else:
                self.fashion_train = fashion_full
                self.fashion_val = fashion_full

        if stage == "test" and not self.test_on_train:
            self.fashion_test = FashionMNIST(self.data_dir, train=False, transform=self.transform)
        elif stage == "test" and self.test_on_train:
            self.fashion_test = FashionMNIST(self.data_dir, train=True, transform=self.transform)

        if stage == "predict" and not self.test_on_train:
            self.fashion_predict = FashionMNIST(self.data_dir, train=False, transform=self.transform)
        elif stage == "predict" and self.test_on_train:
            self.fashion_predict = FashionMNIST(self.data_dir, train=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.fashion_train, batch_size=self.batch_size, pin_memory=self.pin_memory, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.fashion_val, batch_size=self.batch_size, pin_memory=self.pin_memory, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.fashion_test, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.fashion_predict, batch_size=self.batch_size, shuffle=False)

    @property
    def train_dataset(self) -> Dataset:
        return self.fashion_train

    @property
    def val_dataset(self) -> Dataset:
        return self.fashion_val

    @property
    def test_dataset(self) -> Dataset:
        return self.fashion_test

    @property
    def predict_dataset(self) -> Dataset:
        return self.fashion_predict

class DebugFashionMNISTDataModule(LightningDataModuleWrapper):
    def __init__(self,
                 data_dir: str,
                 batch_size: int,
                 num_samples: int,
                 pin_memory: bool = False,
                 num_workers: int = 0,
                 train_set_frac: float = .8,
                 seed: int = 42):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                lambda x: x > 0,
                lambda x: x.type(torch.get_default_dtype())
            ]
        )

        self.num_samples = num_samples
        self.seed = seed
        self.train_set_frac = train_set_frac
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None
        self.mnist_predict = None

    def prepare_data(self):
        # download
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = FashionMNIST(self.data_dir, train=True, transform=self.transform)
            mnist_subset = Subset(dataset=mnist_full, indices=torch.arange(self.num_samples))
            train_set_size = int(self.train_set_frac * len(mnist_subset))
            valid_set_size = len(mnist_subset) - train_set_size
            seed = torch.Generator().manual_seed(self.seed)
            self.mnist_train, self.mnist_val = random_split(mnist_subset, [train_set_size, valid_set_size], generator=seed)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            mnist_test = FashionMNIST(self.data_dir, train=False, transform=self.transform)
            self.mnist_test = Subset(dataset=mnist_test, indices=torch.arange(self.num_samples))

        if stage == "predict":
            mnist_predict = FashionMNIST(self.data_dir, train=False, transform=self.transform)
            self.mnist_predict = Subset(dataset=mnist_predict, indices=torch.arange(self.num_samples))

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    @property
    def train_dataset(self) -> Dataset:
        return self.mnist_train

    @property
    def val_dataset(self) -> Dataset:
        return self.mnist_val

    @property
    def test_dataset(self) -> Dataset:
        return self.mnist_test

    @property
    def predict_dataset(self) -> Dataset:
        return self.mnist_predict

