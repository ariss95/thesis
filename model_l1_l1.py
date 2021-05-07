import data_loader
from sparselandtools.dictionaries import DCTDictionary
import torch
import torch.nn as nn

class l1_l1(nn.Module):
    def __init__(self, input):
        super(first_RNN, self).__init__()
        self.input = input
        self.compressed = int(self.input/4)
        self.hidden_size = self.input * 4
        self.device = torch.device('cpu')
        self.A_matrix = np.asarray(
            np.random.RandomState().uniform( size=(self.compressed, self.input)), dtype=np.float32)
        self.matrix_A = torch.tensor(self.A_matrix, device=self.device, dtype=torch.float32, requires_grad=True)
        dct_dictionary = DCTDictionary(int(sqrt(self.input)),int(sqrt(self.hidden_size)))
        self.Dict_D = torch.tensor(dct_dictionary.matrix, device=self.device, dtype=torch.float32, requires_grad=True)#############FIX
        self.h_0 = torch.zeros((self.batch_size, 2*self.input), device=self.device, dtype=torch.float32,
                               requires_grad=True)
        self.affine_G = torch.eye(self.hidden_size, device=self.device, dtype=torch.float32, requires_grad=True)
        self.l_1 = torch.tensor(1.0, device=self.device, dtype=torch.float32, requires_grad=True)
        self.l_2 = torch.tensor(0.01, device=self.device, dtype=torch.float32, requires_grad=True)
        self.a = torch.tensor(1.0, device=self.device, dtype=torch.float32, requires_grad=True)
        self.hidden_layers = 3

    def activation_func_phi(u, v, l1, l2, a):
        g1 = l1/a
        g2 = l2/a
        temp_tensor = torch.zeros(u.size(), device="cpu")
        condition1 = (((v >= 0) & (v+g1+g2 <= u)) | ((v < 0) & (u >= g1 + g2)))
        temp_tensor[condition1] = u[condition1] - g1 - g2
        condition2 = (((v >= 0) & (v + g1 - g2 <= u) & (u < v + g1 + g2)) |
                    ((v < 0) & (v - g1 + g2 > u) & (u >= v - g1 - g2)))
        temp_tensor[condition2] = v[condition2]
        condition3 = (((v >= 0) & (g1-g2 <= u) & (v+g1-g2 > u))
                    | ((v < 0) & (u < v-g1-g2)))
        temp_tensor[condition3] = u[condition3] - g1 + g2
        condition4 = (((v >= 0) & (u >= -g1-g2) & (u < g1-g2))
                    | ((v < 0) & (u >= -g1 + g2)(u < g1+g2)))
        temp_tensor[condition4] = 0
        condition5 = ((v>=0) & (u<-g1-g2))
        temp_tensor[condition5] = u[condition5] + g1 + g2
        condition6 = ((v<0) & (u<v-g1-g2))
        temp_tensor[condition6] = u[condition5] - g1 + g2
        return temp_tensor

    def forward(self, data):
        U = (1/self.a)*torch.mm(self.matrix_A.t(), self.Dict_D.t())
        AD = torch.mm(self.matrix_A, self.Dict_D)
        ADG = torch.mm(AD, self.affine_G)
        W = self.affine_G - torch.mm(U, ADG)
        S = torch.eye(self.hidden_size, device=self.device, dtype=torch.float32) - torch.mm(U, AD)
        h_previous = self.h_0
        input_ = data#TODO deside shape of input
        for t in range(len(data)):
            for i in range(self.hidden_layers):
                if i==0:
                    u = torch.mm(W, self.h_0) + torch.mm(U, input_[t])
                else:
                    u = torch.mm(S, h_previous) + torch.mm(U, input_[t])
                h_k = activation_func_phi(u, torch.mm(self.affine_G, h_previous), self.l_1, self.l_2, self.a)
                s = torch.mm(self.Dict_D, h_k)