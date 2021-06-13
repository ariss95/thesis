import data_loader
from sparselandtools.dictionaries import DCTDictionary
import torch
import torch.nn as nn
import numpy as np
import time
import math
import plot_utils as pl
from math import sqrt

class l1_l1(nn.Module):
    def __init__(self, input, compression_rate, divider_for_A, K, hidden_size, batch_size):
        super(l1_l1, self).__init__()
        self.input = input
        self.batch_size = batch_size
        self.compressed = int(self.input * compression_rate)
        self.hidden_size = hidden_size #self.input * 4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.A_matrix = np.asarray(np.random.RandomState().uniform( size=(self.compressed, self.input)) / divider_for_A, dtype=np.float32)
        self.matrix_A = torch.tensor(self.A_matrix, device=self.device,  requires_grad=True)
        print(self.matrix_A.size())

        dct_dictionary = DCTDictionary(int(sqrt(self.input)), int(sqrt(self.hidden_size)))
        self.Dict_D = torch.tensor(dct_dictionary.matrix, device=self.device, dtype=torch.float32, requires_grad=True)
        #self.Dict_D = torch.randn(256,1024, device=self.device, requires_grad=True)
        #print(self.Dict_D.max(), self.Dict_D.min())
        self.h_0 = torch.zeros((self.batch_size, self.hidden_size), device=self.device, requires_grad=True)
        self.affine_G = torch.eye(self.hidden_size, device=self.device,  requires_grad=True)
        self.l_1 = torch.tensor(1.0, device=self.device,  requires_grad=True)
        self.l_2 = torch.tensor(0.01, device=self.device, requires_grad=True)
        self.a = torch.tensor(1.0, device=self.device,   requires_grad=True)
        self.hidden_layers = K
        self.parameters = [self.matrix_A, self.Dict_D, self.h_0, self.a, self.l_1, self.l_2]

    def activation_func_phi(self, u, v, l1, l2, a):
        g1 = l1/a
        g2 = l2/a
        temp_tensor = torch.zeros(u.size(), device=self.device)
        condition1 = (((v >= 0) & (v+g1+g2 <= u) &(u< math.inf) ) | ((v < 0) & (u >= g1 + g2) &(u< math.inf)))
        temp_tensor[condition1] = u[condition1] - g1 - g2
        condition2 = (((v >= 0) & (v + g1 - g2 <= u) & (u < v + g1 + g2)) |
                    ((v < 0) & (v - g1 + g2 > u) & (u >= v - g1 - g2)))
        temp_tensor[condition2] = v[condition2]
        condition3 = (((v >= 0) & (g1-g2 <= u) & (v+g1-g2 > u))
                    | ((v < 0) & (u < v-g1-g2)))
        temp_tensor[condition3] = u[condition3] - g1 + g2
        condition4 = (((v >= 0) & (u >= -g1-g2) & (u < g1-g2))
                    | ((v < 0) & (u >= -g1 + g2) & (u < g1+g2)))
        temp_tensor[condition4] = 0
        condition5 = ((v>=0) & (u<-g1-g2) &(u> -math.inf))
        temp_tensor[condition5] = u[condition5] + g1 + g2
        condition6 = ((v<0) & (u<v-g1-g2) & (u> -math.inf))
        temp_tensor[condition6] = u[condition6] - g1 + g2
        #print(temp_tensor[condition1].size(), temp_tensor[condition2].size(),temp_tensor[condition3].size(),temp_tensor[condition4].size(), temp_tensor[condition5].size(), temp_tensor[condition6].size())
        #print(temp_tensor)
        return temp_tensor

    def forward(self, data):
        time_steps = len(data)
        U = (1/self.a)*torch.mm(self.Dict_D.t(), self.matrix_A.t())
        AD = torch.mm(self.matrix_A, self.Dict_D)
        #print(self.Dict_D.shape)
        ADG = torch.mm(AD, self.affine_G)
        W = self.affine_G - torch.mm(U, ADG)
        S = torch.eye(self.hidden_size, device=self.device, dtype=torch.float32) - torch.mm(U, AD)
        h_previous = self.h_0
        s_t = []
        h = []
        input_ = data
        input_ = input_.reshape([-1, self.input])
        compressed_input = torch.mm(input_, self.matrix_A.t())
        compressed_input = compressed_input.view([time_steps, self.batch_size, -1])
       # pl.save_frame(compressed_input, "compressed.png")
        for t in range(time_steps):
            for i in range(self.hidden_layers):
                if i==0:
                    u = torch.mm(self.h_0, W) + torch.mm(compressed_input[t], U.t())
                else:
                    u = torch.mm(h_k, S) + torch.mm(compressed_input[t], U.t())
                h_k = self.activation_func_phi(u, torch.mm(h_previous, self.affine_G), self.l_1, self.l_2, self.a)

            s = torch.mm(h_k, self.Dict_D.t())
            h_previous = h_k
            
            h.append(h_k)
            s_t.append(s)#TODO change to s^t, to avoid reshape
        #print(s_t)
        #time.sleep(12)
        #print("forward111111111111\n")
        output = torch.stack(s_t)
        sparse_representation = torch.stack(h)
        return output
