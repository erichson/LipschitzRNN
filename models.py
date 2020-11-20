import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from copy import deepcopy

from tools import *

import torchdiffeq
#from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint


class LipschitzRNN_ODE(nn.Module):
    "Linear params with forcing"

    def __init__(self, n_units, beta, gamma, init_std):
        super(LipschitzRNN_ODE, self).__init__()
        self.device = get_device()

        self.gamma = gamma
        self.beta = beta

        self.tanh = nn.Tanh()

        self.z = torch.zeros(n_units)
        self.C = nn.Parameter(gaussian_init_(n_units, std=init_std))
        self.B = nn.Parameter(gaussian_init_(n_units, std=init_std))
        self.I = torch.eye(n_units).to(self.device)
        self.i = 0

    def forward(self, t, h):
        if self.i == 0:
            self.A = self.beta * (self.B - self.B.transpose(1, 0)) + (
                1 - self.beta) * (self.B +
                                  self.B.transpose(1, 0)) - self.gamma * self.I
            self.W = self.beta * (self.C - self.C.transpose(1, 0)) + (
                1 - self.beta) * (self.C +
                                  self.C.transpose(1, 0)) - self.gamma * self.I

        return torch.matmul(
            h, self.A) + self.tanh(torch.matmul(h, self.W) + self.z)


class rnn_models(nn.Module):
    def __init__(self,
                 input_dim,
                 output_classes,
                 n_units=128,
                 eps=0.01,
                 beta=0.8,
                 gamma=0.01,
                 gated=False,
                 init_std=1,
                 alpha=1,
                 model='LipschitzRNN',
                 solver='euler'):
        super(rnn_models, self).__init__()

        self.n_units = n_units
        self.eps = eps
        self.model = model
        self.solver = solver
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.E = nn.Linear(input_dim, n_units)
        self.D = nn.Linear(n_units, output_classes)
        self.I = torch.eye(n_units).to(get_device())

        if self.model == 'simpleRNN':
            self.W = nn.Linear(n_units, n_units, bias=False)
            self.W.weight.data = gaussian_init_(n_units, std=init_std)

        elif self.model == 'resRNN':
            self.W = nn.Linear(n_units, n_units, bias=False)
            self.W.weight.data = gaussian_init_(n_units, std=init_std)

        elif self.model == 'asymRNN':
            self.C = nn.Parameter(gaussian_init_(n_units, std=init_std))

        elif self.model == 'calRNN':
            U, _, V = torch.svd(gaussian_init_(n_units, std=init_std))
            self.C = nn.Parameter(torch.mm(U, V.t()).float())

        elif self.model == 'LipschitzRNN':
            self.C = nn.Parameter(gaussian_init_(n_units, std=init_std))
            self.B = nn.Parameter(gaussian_init_(n_units, std=init_std))

        elif self.model == 'LipschitzRNN_gated':
            self.C = nn.Parameter(gaussian_init_(n_units, std=init_std))
            self.B = nn.Parameter(gaussian_init_(n_units, std=init_std))
            self.E_gate = nn.Linear(input_dim, n_units)

        elif self.model == 'LipschitzRNN_ODE':
            self.func = LipschitzRNN_ODE(n_units, beta, gamma, init_std)

        else:
            print("Unexpected model!")
            raise

    def forward(self, x):
        T = x.shape[1]
        h = torch.zeros(x.shape[0], self.n_units).to(which_device(self))

        for i in range(T):
            z = self.E(x[:, i, :])

            if self.model == 'simpleRNN':
                h = self.tanh(self.W(h) + z)

            elif self.model == 'resRNN':
                h = h + self.eps * self.tanh(self.W(h) + z)

            elif self.model == 'asymRNN':
                if i == 0:
                    W = self.C - self.C.transpose(1, 0) - self.gamma * self.I
                h = h + self.eps * self.tanh(torch.matmul(h, W) + z)

            elif self.model == 'calRNN':
                if i == 0:
                    C = self.C - self.C.transpose(1, 0)
                    W = torch.matmul(torch.inverse(self.I + C), self.I - C)
                h = self.tanh(torch.matmul(h, W) + z)

            elif self.model == 'LipschitzRNN':
                if i == 0:
                    A = self.beta * (self.B - self.B.transpose(1, 0)) + (
                        1 - self.beta) * (self.B + self.B.transpose(
                            1, 0)) - self.gamma * self.I
                    W = self.beta * (self.C - self.C.transpose(1, 0)) + (
                        1 - self.beta) * (self.C + self.C.transpose(
                            1, 0)) - self.gamma * self.I
                h = h + self.eps * self.alpha * torch.matmul(
                    h, A) + self.eps * self.tanh(torch.matmul(h, W) + z)

            elif self.model == 'LipschitzRNN_gated':
                if i == 0:
                    A = self.beta * (self.B - self.B.transpose(1, 0)) + (
                        1 - self.beta) * (self.B + self.B.transpose(
                            1, 0)) - self.gamma * self.I
                    W = self.beta * (self.C - self.C.transpose(1, 0)) + (
                        1 - self.beta) * (self.C + self.C.transpose(
                            1, 0)) - self.gamma * self.I
                z_gate = self.E_gate(x[:, i, :])
                Wh = torch.matmul(h, W)
                Ah = torch.matmul(h, A)
                q1 = self.alpha * Ah + self.tanh(Wh + z)
                q2 = self.sigmoid(Wh + z_gate)
                h = h + self.eps * q1 * q2

            elif self.model == 'LipschitzRNN_ODE':
                self.func.z = z
                self.func.i = i
                h = odeint(self.func,
                           h,
                           torch.tensor([0, self.eps]).float(),
                           method=self.solver)[-1, :, :]

        # Decoder
        #----------
        out = self.D(h)
        return out
