import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

import os
import argparse
import pickle

from tools import * 
from get_data import *
from models import *


#==============================================================================
# Training settings
#==============================================================================

parser = argparse.ArgumentParser(description='MNIST Example')
#
parser.add_argument('--name', type=str, default='mnist', metavar='N', help='dataset')
#
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
#
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
#
parser.add_argument('--epochs', type=int, default=38, metavar='N', help='number of epochs to train (default: 90)')
#
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
#
parser.add_argument('--lr_decay', type=float, default=0.1, help='learning rate decay value (default: 0.1)')
#
parser.add_argument('--lr_decay_epoch', type=int, nargs='+', default=[30, 60, 80], help='decrease learning rate at these epochs.')
#
parser.add_argument('--wd', default=0.0, type=float, metavar='W', help='weight decay (default: 0.0)')
#
parser.add_argument('--gamma', default=0.001, type=float, metavar='W', help='diffiusion rate')
#
parser.add_argument('--beta', default=0.7, type=float, metavar='W', help='skew level')
#
parser.add_argument('--model', type=str, default='LipschitzRNN', metavar='N', help='model name')
#
parser.add_argument('--n_units', type=int, default=128, metavar='S', help='number of hidden units')
#
parser.add_argument('--eps', default=0.05, type=float, metavar='W', help='time step for euler scheme')
#
parser.add_argument('--gated', default=False, type=bool, metavar='W', help='gated')
#
parser.add_argument('--T', default=98, type=int, metavar='W', help='time steps')
#
parser.add_argument('--init_std', type=float, default=6, metavar='S', help='control of std for initilization')
#
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 0)')
#
parser.add_argument('--gclip', type=int, default=10, metavar='S', help='gradient clipping')
#
parser.add_argument('--optimizer', type=str, default='SGD', metavar='N', help='optimizer')
#
parser.add_argument('--alpha', type=float, default=1, metavar='S', help='for ablation study')
#
args = parser.parse_args()

if not os.path.isdir(args.name + '_results'):
    os.mkdir(args.name + '_results')


#==============================================================================
# set random seed to reproduce the work
#==============================================================================
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

#==============================================================================
# get device
#==============================================================================
device = get_device()

#==============================================================================
# get dataset
#==============================================================================
if args.name == 'mnist':
    train_loader, test_loader = getData(name='mnist', train_bs=args.batch_size, test_bs=args.test_batch_size)  
    model = rnn_models(input_dim=int(784/args.T), output_classes=10, n_units=args.n_units, 
                 eps=args.eps, beta=args.beta, gamma=args.gamma, gated=args.gated,
                 model=args.model, init_std=args.init_std, alpha=args.alpha).to(device)
            
elif args.name == 'pmnist':
    train_loader, test_loader = getData(name='pmnist', train_bs=args.batch_size, test_bs=args.test_batch_size)  
    model = rnn_models(input_dim=int(784/args.T), output_classes=10, n_units=args.n_units, 
                 eps=args.eps, beta=args.beta, gamma=args.gamma, gated=args.gated,
                 model=args.model, init_std=args.init_std, alpha=args.alpha).to(device)    
    
elif args.name == 'cifar10':    
    train_loader, test_loader = getData(name='cifar10', train_bs=args.batch_size, test_bs=args.test_batch_size)          
    model = rnn_models(input_dim=int(1024/args.T*3), output_classes=10, n_units=args.n_units, 
                 eps=args.eps, beta=args.beta, gamma=args.gamma, gated=args.gated,
                 model=args.model, init_std=args.init_std).to(device) 

elif args.name == 'cifar10_noise':  
    train_loader, test_loader = getData(name='cifar10', train_bs=args.batch_size, test_bs=args.test_batch_size)              
    model = rnn_models(input_dim=int(1024/args.T*3), output_classes=10, n_units=args.n_units, 
                 eps=args.eps, beta=args.beta, gamma=args.gamma, gated=args.gated,
                 model=args.model, init_std=args.init_std).to(device)     
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)        
    noise = torch.randn(1,968,32,3).float()

#==============================================================================
# Model summary
#==============================================================================
print(model)    
print('**** Setup ****')
print('Total params: %.2fk' % (sum(p.numel() for p in model.parameters())/1000.0))
print('************')    
   

if args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
elif  args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
else:
    print("Unexpected optimizer!")
    raise 


loss_func = nn.CrossEntropyLoss().to(device)

# training and testing
count = 0
loss_hist = []
max_eig_A = []
max_eig_W = []
test_acc = []

for epoch in range(args.epochs):
    model.train()
    lossaccum = 0
    
    for step, (x, y) in enumerate(train_loader):
        count += 1
        
        # Reshape data for recurrent unit
        if args.name == 'mnist' or args.name == 'pmnist':
            inputs = Variable(x.view(-1, args.T, int(784/args.T))).to(device) # reshape x to (batch, time_step, input_size)
            targets = Variable(y).to(device)
            
        elif args.name == 'cifar10':            
            inputs = Variable(x.view(-1, args.T, int(1024/args.T*3))).to(device) # reshape x to (batch, time_step, input_size)
            targets = Variable(y).to(device)   

        elif args.name == 'cifar10_noise':
            x = x.view(-1, 32, int(96))
            x = torch.cat((x, noise.repeat(x.shape[0],1,1,1).view(-1, 968, int(96))), 1) # reshape x to (batch, time_step, input_size)
            inputs = Variable(x).to(device)             
            targets = Variable(y).to(device)   

                 
        # send data to recurrent unit    
        output = model(inputs)   
        loss = loss_func(output, targets)
        
        
        optimizer.zero_grad()
        loss.backward()                 
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gclip) # gradient clip
        optimizer.step() # update weights
        lossaccum += loss.item()

    loss_hist.append(lossaccum)    
     
    if epoch % 1 == 0:
        model.eval()
        correct = 0
        total_num = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            
            if args.name == 'mnist' or args.name == 'pmnist':
                output = model(data.view(-1, args.T, int(784/args.T)))
            
            elif args.name == 'cifar10':             
                output = model(data.view(-1, args.T, int(1024/args.T*3)))
            
            elif args.name == 'cifar10_noise':
                data = data.view(-1, 32, int(96))
                data = torch.cat((data, noise.repeat(data.shape[0],1,1,1).view(-1, 968, int(96))), 1) # reshape x to (batch, time_step, input_size)
                data = Variable(data).to(device)             
                output = model(data)
            
            
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total_num += len(data)
        
        accuracy = correct / total_num
        test_acc.append(accuracy)
        print('Epoch: ', epoch, 'Iteration: ', count, '| train loss: %.4f' % loss.item(), '| test accuracy: %.3f' % accuracy)

        if args.model == 'LipschitzRNN':
            B = model.B.data.cpu().numpy()            
            A = args.beta * (B - B.T) + (1-args.beta) * (B + B.T) - args.gamma*np.eye(args.n_units)
            
            e, _ = np.linalg.eig(A)
            print(np.max(e.real))
            max_eig_A.append(np.max(e.real))
            
            C = model.C.data.cpu().numpy()            
            W = args.beta * (C - C.T) + (1-args.beta) * (C + C.T) - args.gamma*np.eye(args.n_units)
            e, _ = np.linalg.eig(W)
            print(np.max(e.real))            
            max_eig_W.append(np.max(e.real))

        elif args.model == 'LipschitzRNN_ODE':
            B = model.func.B.data.cpu().numpy()            
            A = args.beta * (B - B.T) + (1-args.beta) * (B + B.T) - args.gamma*np.eye(args.n_units)
            e, _ = np.linalg.eig(A)
            print(np.max(e.real))
            max_eig_A.append(np.max(e.real))
            
            C = model.func.C.data.cpu().numpy()            
            W = args.beta * (C - C.T) + (1-args.beta) * (C + C.T) - args.gamma*np.eye(args.n_units)
            e, _ = np.linalg.eig(W)
            print(np.max(e.real))            
            max_eig_W.append(np.max(e.real))
                                    
        elif args.model == 'resRNN':
            D = model.C.weight.data.cpu().numpy()            
            e, _ = np.linalg.eig(D)
            print(np.max(e.real))     
            
        elif args.model == 'asymRNN':
            D = model.C.data.cpu().numpy()
            W = (D - D.T) - args.gamma*np.eye(args.n_units)
            e, _ = np.linalg.eig(W)
            print(np.max(e.real))              

    # schedule learning rate decay    
    optimizer=exp_lr_scheduler(epoch, optimizer, decay_eff=args.lr_decay, decayEpoch=args.lr_decay_epoch)

#torch.save(model.state_dict(), args.name + '_results/' + args.model + '_' + str(args.T) + str(args.n_units) +'.pkl')  
torch.save(model, args.name + '_results/' + args.model + '_' + args.name + '_T' 
           + str(args.T) + '_units' + str(args.n_units) + '_beta' + str(args.beta) 
           + '_gamma' + str(args.gamma) + '_eps' + str(args.eps) 
           + '_init' + str(args.init_std) + '_init' + str(args.init_std) 
           + '_seed' + str(args.seed) + '.pkl')  



data = {'loss': lossaccum, 'eigA': max_eig_A, 'eigW': max_eig_W, 'testacc': test_acc}
f = open(args.name + '_results/' + args.model + '_' + args.name + '_T' 
           + str(args.T) + '_units' + str(args.n_units) + '_beta' + str(args.beta) 
           + '_gamma' + str(args.gamma) + '_eps' + str(args.eps) 
           + '_init' + str(args.init_std) + '_init' + str(args.init_std) 
           + '_seed' + str(args.seed) + '_loss.pkl',"wb")
pickle.dump(data,f)
f.close()