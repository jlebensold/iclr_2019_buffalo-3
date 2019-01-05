import fire
import torch
import numpy as np

from torch.utils import data
from torchvision import datasets, transforms
from typing import Tuple, Dict, Callable
from pathlib import Path

from torch_harness import TorchHarness
from baseline_models import ReLUNet, TanhNet, MaxoutNet
from dilation_erosion import DenMoNet

data_dir = Path(__file__).parent / 'data'
model_dir = Path(__file__).parent / 'models'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Datasets

def load_cifar10() -> Tuple[data.Dataset, data.Dataset, int]:
    """Load CIFAR-10 train, test, and size."""
    trans = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.CIFAR10(str(data_dir), train=True, download=True, transform=trans)
    test_set = datasets.CIFAR10(str(data_dir), train=False, download=True, transform=trans)
    return train_set, test_set, 32 * 32 * 3


def load_fashion_mnist() -> Tuple[data.Dataset, data.Dataset, int]:
    """Load fashion MNIST train, test, and size."""
    trans = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.FashionMNIST(str(data_dir), train=True, download=True, transform=trans)
    test_set = datasets.FashionMNIST(str(data_dir), train=False, download=True, transform=trans)
    return train_set, test_set, 28 * 28


def load_mnist() -> Tuple[data.Dataset, data.Dataset, int]:
    """Load MNIST train, test, and size."""
    trans = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(str(data_dir), train=True, download=True, transform=trans)
    test_set = datasets.MNIST(str(data_dir), train=False, download=True, transform=trans)
    return train_set, test_set, 28 * 28


def load_circles() -> Tuple[data.Dataset, data.Dataset, int]:
    """Load circles train, test, and size."""
    training = np.load('data/circle_training.npy')
    test = np.load('data/circle_test.npy')
    train_set = data.TensorDataset(
        torch.FloatTensor(training[:, :-1]),
        torch.LongTensor(training[:, -1])
    )

    test_set = data.TensorDataset(
        torch.FloatTensor(test[:, :-1]),
        torch.LongTensor(test[:, -1])
    )

    return train_set, test_set, 2


def load_squares() -> Tuple[data.Dataset, data.Dataset, int]:
    """Load squares train, test, and size."""
    x = np.load('data/coordinates/squares_features.npy')
    y = np.load('data/coordinates/squares_labels.npy')
    train_set = data.TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    test_set = data.TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return train_set, test_set, 2


dataset_map: Dict[str, Callable] = {
    "cifar10": load_cifar10,
    "fashion_mnist": load_fashion_mnist,
    "mnist": load_mnist,
    "circles": load_circles,
    "squares": load_squares
}

# Models

baseline_model_map: Dict[str, Callable] = {
    'relu': ReLUNet,
    'tanh': TanhNet,
    'maxout': MaxoutNet
}


def run_and_save_model(model, train, test, epochs, name):
    harness = TorchHarness(model, model.name(name), train, test, epochs=epochs)
    harness.train_and_evaluate()
    model.store(name, model_dir)


# Runners

def run_denmo(dset_name: str, erosions: int = 5, dilations: int = 5, epochs: int = 2):
    """Run denmo on a dataset.

    Datasets:
        * mnist
        * fashion_mnist
        * cifar10
    """
    train, test, size = dataset_map[dset_name]()
    model = DenMoNet(size, dilations, erosions, 10)
    run_and_save_model(model, train, test, epochs, dset_name)


def run_baseline(model_name: str, dset_name: str, h_layers: int = 200, epochs: int = 2):
    """Run a baseline model on a dataset.

    Datasets:
        * mnist
        * fashion_mnist
        * cifar10

    Models:
        * relu
        * tanh
        * maxout
    """
    train, test, size = dataset_map[dset_name]()
    model = baseline_model_map[model_name](size, h_layers)
    run_and_save_model(model, train, test, epochs, dset_name)


def predict_coordinates(path: str):
    # train, test, size = dataset_map[dset_name]()
    model = torch.load(path )
    model.eval()
    features = np.load('data/coordinates/coordinates.npy')
    coordinates = torch.from_numpy(features).to(device).float()
    predictions = model(coordinates).data.cpu().numpy()
    classes = predictions.argmax(axis=1)

    zero_coordinates = np.where(classes == 0)[0]
    one_coordinates = np.where(classes == 1)[0]

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(features[zero_coordinates, 0], features[zero_coordinates, 1], linestyle='None', marker='o')
    ax.plot(features[one_coordinates, 0], features[one_coordinates, 1], linestyle='None', marker='o')
    plt.show()


if __name__ == '__main__':
    fire.Fire()
