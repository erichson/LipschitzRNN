import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import pickle

from tools import *

from torch.utils.data import Subset
from torch._utils import _accumulate


def getData(name='cifar10', train_bs=128, test_bs=1000):

    if name == 'mnist':

        train_loader = datasets.MNIST('./data',
                                      train=True,
                                      download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                      ]))

        val_loader = datasets.MNIST('./data',
                                    train=True,
                                    download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]))

        offset = 3000
        rng = np.random.RandomState(1234)
        R = rng.permutation(len(train_loader))
        lengths = (len(train_loader) - offset, offset)
        train_loader, val_loader = [
            Subset(train_loader, R[offset - length:offset])
            for offset, length in zip(_accumulate(lengths), lengths)
        ]

        train_loader = torch.utils.data.DataLoader(train_loader,
                                                   batch_size=train_bs,
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_loader,
                                                 batch_size=test_bs,
                                                 shuffle=False)

        test_loader = torch.utils.data.DataLoader(datasets.MNIST(
            './data',
            train=False,
            download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])),
                                                  batch_size=test_bs,
                                                  shuffle=False)

    if name == 'pmnist':

        trainset = datasets.MNIST(root='./data',
                                  train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                  ]))

        testset = datasets.MNIST(root='./data',
                                 train=False,
                                 download=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                 ]))

        x_train = trainset.train_data
        y_train = trainset.targets

        x_test = testset.test_data
        y_test = testset.targets

        torch.manual_seed(12008)
        perm = torch.randperm(784)

        x_train_permuted = x_train.reshape(x_train.shape[0], -1)
        x_train_permuted = x_train_permuted[:, perm]
        x_train_permuted = x_train_permuted.reshape(x_train.shape[0], 28, 28)

        x_test_permuted = x_test.reshape(x_test.shape[0], -1)
        x_test_permuted = x_test_permuted[:, perm]
        x_test_permuted = x_test_permuted.reshape(x_test.shape[0], 28, 28)

        x_train_permuted = add_channels(x_train_permuted)
        x_test_permuted = add_channels(x_test_permuted)

        train_loader = torch.utils.data.TensorDataset(x_train_permuted.float(),
                                                      y_train)

        offset = 3000
        rng = np.random.RandomState(1234)
        R = rng.permutation(len(train_loader))
        lengths = (len(train_loader) - offset, offset)
        train_loader, val_loader = [
            Subset(train_loader, R[offset - length:offset])
            for offset, length in zip(_accumulate(lengths), lengths)
        ]

        train_loader = torch.utils.data.DataLoader(train_loader,
                                                   batch_size=train_bs,
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_loader,
                                                 batch_size=test_bs,
                                                 shuffle=False)

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_test_permuted.float(), y_test),
            batch_size=test_bs,
            shuffle=False)

    if name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_loader = datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform_train)

        offset = 3000
        rng = np.random.RandomState(1234)
        R = rng.permutation(len(train_loader))
        lengths = (len(train_loader) - offset, offset)
        train_loader, val_loader = [
            Subset(train_loader, R[offset - length:offset])
            for offset, length in zip(_accumulate(lengths), lengths)
        ]

        train_loader = torch.utils.data.DataLoader(train_loader,
                                                   batch_size=train_bs,
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_loader,
                                                 batch_size=test_bs,
                                                 shuffle=False)

        testset = datasets.CIFAR10(root='./data',
                                   train=False,
                                   download=False,
                                   transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=test_bs,
                                                  shuffle=False)

    if name == 'double_pendulum':
        # open a file, where you stored the pickled data
        file = open("./data/double_pendulum.pkl", 'rb')
        data = pickle.load(file)
        file.close()

        trainset = []
        train_target = []
        testset = []
        test_target = []

        for i in range(1, 400):
            trainset.append(data[i:i + 1000])
            train_target.append(data[i + 1000 + 1])

        for i in range(1501, 3000):
            testset.append(data[i:i + 1000])
            test_target.append(data[i + 1000 + 1])

        trainset = np.asarray(trainset)
        testset = np.asarray(testset)
        train_target = np.asarray(train_target)
        test_target = np.asarray(test_target)

        trainset = torch.tensor(trainset)
        testset = torch.tensor(testset)
        train_target = torch.tensor(train_target)
        test_target = torch.tensor(test_target)

        #trainset = add_channels(trainset)
        #testset = add_channels(testset)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(trainset.float(),
                                           train_target.float()),
            batch_size=train_bs,
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(testset.float(),
                                           test_target.float()),
            batch_size=test_bs,
            shuffle=False)

    return train_loader, test_loader, val_loader
