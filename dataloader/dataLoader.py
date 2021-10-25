# data loading functionality
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt


class dataLoader:
    def __init__(self) -> None:
        self.trainingData = datasets.CIFAR100(
            root="../data",
            train=True,
            download=True,
            transform=ToTensor()
        )
        self.testData = datasets.CIFAR100(
            root="../data",
            train=False,
            download=True,
            transform=ToTensor()
        )

        # create dataloaders around the data, make them iterable
        batch_size = 64
        self.trainDataLoader = DataLoader(self.trainingData, batch_size=batch_size)
        self.testDataLoader = DataLoader(self.testData, batch_size=batch_size)   

    


    # test data
    for X, y in test_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break