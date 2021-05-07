import numpy as np
import matplotlib.pyplot as plt
from sparselandtools.dictionaries import DCTDictionary
from math import sqrt
#import plot_utils
import torch


# this class is used for loading the dataset from the .npy file
class Moving_MNIST_Loader:
    def __init__(self, path, time_steps, flatten=True):

        self.data = np.load(path).astype('float32')

        if time_steps < self.data.shape[0]:
            self.data = self.data[:time_steps]
        self.num_frames, self.num_samples, self.size = self.data.shape[
            0], self.data.shape[1], self.data.shape[2:]

        if flatten:
            self.data = self.data.reshape(
                [self.num_frames, self.num_samples, -1])

        self.train_set_size = int(self.num_samples * 0.8)
        self.validation_size = self.train_set_size + \
            int(self.num_samples * 0.1)
        self.train = self.data[:, :self.train_set_size, ...]
        self.validate = self.data[:,
                                  self.train_set_size: self.validation_size, ...]
        self.test = self.data[:, self.validation_size:, ...]
        self.train_index = 0
        self.validation_index = 0
        self.testing_index = 0

        print("loading of moving MNIST completed")

    def shuffle(self):
        indices = np.random.permutation(self.train_set_size)
        self.train = self.train[:, indices, ...]

    def get_batch(self, set, batch_size):
        if set == "train":
            if self.train_index + batch_size - 1 >= self.train_set_size:
                self.shuffle()
                self.train_index = 0

            batch = self.train[:,
                               self.train_index:self.train_index + batch_size, ...]
            self.train_index += batch_size
            return batch
        elif set == "test":
            batch = self.test[:,
                              self.testing_index: self.testing_index + batch_size, ...]
            self.testing_index += batch_size
            return batch
        else:
            if self.validation_index + batch_size - 1 >= self.validate.shape[1]:
                self.validation_index = 0
                return []
            batch = self.validate[:,
                                  self.validation_index: self.validation_index + batch_size, ...]
            self.validation_index += batch_size
            return batch
'''
REMOVEEEE!!!!!!!!!!
# TODO REMOVE
        self.input = 256
        self.compressed = int(self.input/4)
        self.hidden_size = self.input * 4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.A_matrix = np.asarray(
            np.random.RandomState().uniform(size=(self.compressed, self.input)))
        self.matrix_A = torch.tensor(
            self.A_matrix, device=self.device, dtype=torch.float32, requires_grad=True)
        dct_dictionary = DCTDictionary(
            int(sqrt(self.input)), int(sqrt(self.hidden_size)))
        self.Dict_D = torch.tensor(
            dct_dictionary.matrix, device=self.device, dtype=torch.float32, requires_grad=True)  # FIX
        self.h_0 = torch.zeros((3, 4*self.input), device=self.device, dtype=torch.float32,
                               requires_grad=True)
        self.affine_G = torch.eye(
            self.hidden_size, device=self.device, dtype=torch.float32, requires_grad=True)
        self.l_1 = torch.tensor(1.0, device=self.device,
                                dtype=torch.float32, requires_grad=True)
        self.l_2 = torch.tensor(0.01, device=self.device,
                                dtype=torch.float32, requires_grad=True)
        self.a = torch.tensor(1.0, device=self.device,
                              dtype=torch.float32, requires_grad=True)
        self.hidden_layers = 3

def activation_func_phi(u, v, l1, l2, a):
    print(u.size())
    g1 = l1/a
    g2 = l2/a
    temp_tensor = torch.zeros(u.size(), device=self.device)
    condition1 = (((v >= 0) & (v+g1+g2 <= u)) | ((v < 0) & (u >= g1 + g2)))
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
    condition5 = ((v>=0) & (u<-g1-g2))
    temp_tensor[condition5] = u[condition5] + g1 + g2
    condition6 = ((v<0) & (u<v-g1-g2))
    temp_tensor[condition6] = u[condition6] - g1 + g2
    return temp_tensor


path = '/home/aris/Desktop/anomaly_detection/movingMnist/mnist_test_seq_16.npy'
time_steps = 20
data_loader = Moving_MNIST_Loader(
    path=path, time_steps=time_steps, flatten=True)

data = data_loader.get_batch("train", 3)
A_matrix = np.asarray(
    np.random.RandomState().uniform(size=(64, 256)))
matrix_A = torch.tensor(data_loader.A_matrix,  dtype=torch.float32)
dct_dictionary = DCTDictionary(int(sqrt(256)), int(sqrt(1024)))
print(dct_dictionary.matrix.shape)

Dict_D = torch.tensor(dct_dictionary.matrix, dtype=torch.float32)  # FIX


U = (1)*torch.mm(Dict_D.t(), matrix_A.t())
AD = torch.mm(matrix_A, Dict_D)
ADG = torch.mm(AD, data_loader.affine_G)
W = data_loader.affine_G - torch.mm(U, ADG)
S = torch.eye(1024) - torch.mm(U, AD)
h_previous = data_loader.h_0
# input_ = data#TODO deside shape of input
input_ = torch.tensor(data, dtype=torch.float32, device="cpu")
input_ = input_.view([-1, 256])
compressed_input = torch.mm(input_, matrix_A.t())
print(compressed_input.size())

compressed_input = compressed_input.view([time_steps, 3, -1])
print(compressed_input.size())
# for t in range(time_steps):
#input_ = torch.mm(matrix_A, input_[t])

for t in range(1):
    for i in range(1):
        if i == 0:
            temp1 = torch.mm(data_loader.h_0, W)
            print(temp1.size())
            u = temp1 + torch.mm(compressed_input[t], U.t())
        else:
            u = torch.mm(h_previous, S) + torch.mm(compressed_input[t], U.t())
        h_k = activation_func_phi(u, torch.mm(
            h_previous, data_loader.affine_G), data_loader.l_1, data_loader.l_2, data_loader.a)
'''