import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable



def getData(name='cifar10', train_bs=128, test_bs=1000):    
    
   
    
    if name == 'mnist':

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=train_bs, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=test_bs, shuffle=False)


    if name == 'pmnist':

        
        trainset = datasets.MNIST(root='../data',
                                    train=True,
                                    download=True,
                                    transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        
        testset = datasets.MNIST(root='../data',
                                  train=True,
                                  download=True,transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        
    
        rng = np.random.RandomState(1234)
    


        x_train = trainset.train_data
        y_train = trainset.targets
        
        x_test = testset.test_data
        y_test = testset.targets

        #plt.figure()
        #plt.imshow(x_train[0,:,:].data.cpu().numpy())        

        torch.manual_seed(42)
        permuted_idx_row = torch.randperm(x_train.shape[1])
        permuted_idx_col = torch.randperm(x_train.shape[2])

        x_train_permuted = x_train[:, permuted_idx_row, :]
        x_test_permuted = x_test[:, permuted_idx_row, :]

        x_train_permuted = x_train_permuted[:, :, permuted_idx_col]
        x_test_permuted = x_test_permuted[:, :, permuted_idx_col]        
        
        #plt.figure()
        #plt.imshow(x_train_permuted[0,:,:].data.cpu().numpy())        
        
        x_train_permuted = add_channels(x_train_permuted)
        x_test_permuted = add_channels(x_test_permuted)
        
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_permuted.float(), y_train),
                                                  batch_size=train_bs,
                                                  shuffle=True)
        
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_permuted.float(), y_test),
                                                batch_size=test_bs,
                                                shuffle=False)
      
    
    
    
    if name == 'cifar10':
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

        trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True)

        testset = datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False)
    
    


        
    return train_loader, test_loader







