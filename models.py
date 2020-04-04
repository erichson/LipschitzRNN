import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from copy import deepcopy

from tools import *



class relaxRNN(nn.Module):
    def __init__(self, input_dim, output_classes, n_units=64, eps=0.01, beta=0.5, gamma=0.0, gating=False, init_std=1, model = 'relaxRNN'):
        super(relaxRNN, self).__init__()

        self.device = get_device()

        self.n_units = n_units
        self.eps = eps
        self.model = model
        self.gamma = gamma
        self.beta = beta
        self.gating = gating
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.E = nn.Linear(input_dim, n_units)
        self.D = nn.Linear(n_units, output_classes)     
        

        
        if gating == True:
            self.E_gate = nn.Linear(input_dim, n_units)      

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)               
                
        if self.model == 'RNN':
            self.C = nn.Linear(n_units, n_units, bias=False)
            self.C.weight.data = gaussian_init_(self.C.weight.data , std=1)
            
        elif self.model == 'resRNN':
            self.C = nn.Linear(n_units, n_units, bias=False)
            self.C.weight.data = gaussian_init_(self.C.weight.data , std=1)
          
        elif self.model == 'asymRNN':            
            self.C = nn.Parameter(gaussian_init_(n_units, std=init_std))            

        elif self.model == 'calRNN':            
            self.C = nn.Parameter(gaussian_init_(n_units, std=init_std))            
            
        elif self.model == 'relaxRNN':            
            self.C = nn.Parameter(gaussian_init_(n_units, std=init_std))            
            self.B = nn.Parameter(gaussian_init_(n_units, std=init_std))            

        elif self.model == 'ablation':            
            self.C = nn.Parameter(gaussian_init_(n_units, std=init_std))            
          
        else:
            print("Unexpected model!")
            raise           
                                    
 


    def forward(self, x):
        T = x.shape[1]
        h = torch.zeros(x.shape[0], self.n_units).to(self.device)
        
        for i in range(T):
            # Encoder
            z = self.E(x[:,i,:])
                
            if self.model == 'RNN':
                h = self.tanh(self.C(h) + z)                 
                
            elif self.model == 'resRNN':
                h = self.h + self.eps * self.tanh(self.C(h) + z) 
                
            elif self.model == 'asymRNN':
                I = torch.eye(self.n_units).to(self.device)                                
                A = self.C - self.C.transpose(1, 0) - self.gamma * I
                h = h + self.eps * self.tanh(torch.matmul(h, A) + z) 

            elif self.model == 'calRNN':
                I = torch.eye(self.n_units).to(self.device)                
                A = self.C - self.C.transpose(1, 0)
                Q = torch.matmul(torch.inverse(I + A), I - A)                            
                h = self.tanh(torch.matmul(h, Q) + z) 

            elif self.model == 'relaxRNN':
                I = torch.eye(self.n_units).to(self.device)                
                A = self.beta * (self.C - self.C.transpose(1, 0)) + (1-self.beta) * (self.C + self.C.transpose(1, 0)) - self.gamma * I
                D = self.beta * (self.B - self.B.transpose(1, 0)) + (1-self.beta) * (self.B + self.B.transpose(1, 0)) - self.gamma * I
                
                h = h + self.eps * torch.matmul(h, D) + self.eps * self.tanh(torch.matmul(h, A) + z)
                    
            elif self.model == 'ablation':
                I = torch.eye(self.n_units).to(self.device)                                
                A = self.beta * (self.C - self.C.transpose(1, 0)) + (1-self.beta) * (self.C + self.C.transpose(1, 0)) - self.gamma * I
                h_temp = torch.matmul(h, A)
                if self.gating == False: 
                    h = h + self.eps * self.tanh(h_temp + z)
                    
                elif self.gating == True: 
                    z_gate = self.E_gate(x[:,i,:])                    
                    q1 = self.tanh(h_temp + z)                  
                    q2 = self.sigmoid(h_temp + z_gate)                   
                    h = h + self.eps * q1 * q2


        # Decoder 
        #----------
        out = self.D(h)
        
        return out
    
