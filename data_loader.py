import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import random
import torch


# this class is used for loading the dataset from the .npy file
# moving Mnist dataset with 16*16 dimensions found on: https://1drv.ms/u/s!ApHn770BvhH2cJyQS9lzhfiwReA?e=fPQy59 
class Moving_MNIST_Loader:
    def __init__(self, path, time_steps, flatten=True):

        self.data = np.load(path).astype('float32')

        if time_steps < self.data.shape[0]:
            self.data = self.data[:time_steps]
        self.num_frames, self.num_samples, self.size = self.data.shape[
            0], self.data.shape[1], self.data.shape[2:]

        if flatten:
            self.data = self.data.reshape([self.num_frames, self.num_samples, -1])

        self.train_set_size = int(self.num_samples * 0.8)
        self.validation_size = self.train_set_size + \
        int(self.num_samples * 0.1)
        print(self.validation_size)
        self.train = self.data[:, :self.train_set_size, ...]
        self.validate = self.data[:,self.train_set_size: self.validation_size, ...]
        self.test = self.data[:, self.validation_size:, ...]
        self.train_index = 0
        self.validation_index = 0
        self.testing_index = 0
        self.anomaly_index = 0
        self.anomaly_test = []
        self.labels = []

        print("loading of moving MNIST completed")
        print(self.data.shape)

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
            if self.validation_index + batch_size - 1 >= self.validate.shape[1]:
                self.validation_index = 0
            batch = self.test[:,
                              self.testing_index: self.testing_index + batch_size, ...]
            self.testing_index += batch_size
            return batch
        elif set == "validation":
            if self.validation_index + batch_size - 1 >= self.validate.shape[1]:
                self.validation_index = 0
               # return []
            batch = self.validate[:, self.validation_index: self.validation_index + batch_size, ...]
            self.validation_index += batch_size
            return batch
        else:
            if self.anomaly_index == 0:
                self.get_anomaly_test_set()
            #if self.anomaly_index + batch_size > int(self.num_samples * 0.1)
            #    batch_size = int(self.num_samples * 0.1) - self.anomaly_index #TODO FIX batch size and 1000 test set
            batch = self.test[:, self.anomaly_index: self.anomaly_index + batch_size, ...]
            labels_of_batch = self.labels[:,self.anomaly_index: self.anomaly_index + batch_size]
            self.anomaly_index += batch_size
            return batch, labels_of_batch

    def get_anomaly_test_set(self):
        anomaly_test_set = []
        size = int(self.num_samples * 0.1)
        print(size)
        labels = np.zeros((20, size))
        for i in range(size):
            corrupted_frame = self.create_anomaly(i)
            labels[corrupted_frame][i] = 1
        self.labels = labels

    def create_anomaly(self, index):
        rand_sequence = random.randint(0, self.num_samples-1)
        anomaly = self.test[:, index, ...]
       
        corrupted = random.randint(0, 19)
        #frame = anomaly[corrupted].reshape(16,16)
        #break_flag = False
        #the following loop draws a black 3*3 square in the random frame over some non black pixels
        #for corrupted in range(20):
        break_flag = False
        frame = anomaly[corrupted].reshape(16,16)              
        for i in range(1, 14):
            for j in range(1, 14):
                if frame[i][j]!=0 and frame[i][j+1]!=0 and frame[i+2][j]!=0 and frame[i][j+2]!=0 and frame[i+1][j+1]!=0 and frame[i+1][j]!=0 and frame[i+1][j+2]!=0 and frame[i+2][j+2]!=0:
                    frame[i][j]=0
                    frame[i][j+1]=0
                    frame[i][j+2]=0
                    frame[i+1][j]=0
                    frame[i+1][j+1]=0
                    frame[i+1][j+2]=0
                    frame[i+2][j+1]=0
                    frame[i+2][j]=0
                    frame[i+2][j+2]=0
                    
                    break_flag = True
                    break
                if frame[i][j]==0 and frame[i][j+1]==0 and frame[i+2][j]==0 and frame[i][j+2]==0 and frame[i+1][j+1]==0 and frame[i+1][j]==0 and frame[i+1][j+2]==0 and frame[i+2][j+2]==0:
                    frame[i][j]=255
                    frame[i][j+1]=255
                    frame[i][j+2]=255
                    frame[i+1][j]=255
                    frame[i+1][j+1]=255
                    frame[i+1][j+2]=255
                    frame[i+2][j+1]=255
                    frame[i+2][j]=255
                    frame[i+2][j+2]=255
                    #break_flag = True
                    break
            if break_flag:
                break

        #print(corrupted)
        '''
        for i in range(20):
            x1 = anomaly[i].reshape(16, 16)
            plt.figure(1)
            plt.axis("off")
            plt.clf()
            plt.title('output')
            plt.imshow(x1, cmap="gray")
            plt.draw()
            plt.pause(1)
            print(i)
        '''
        return corrupted

class UCSD_loader:
    def __init__(self, path, flatten=True):

        self.train = np.load(path + "Dataset.v1p2.npy").astype('float32')
        self.test = np.load(path + "Dataset_TestSet.npy").astype('float32')
        self.labels = np.load(path + "labels.npy").astype('int32')
        self.num_frames, self.num_train_samples, self.size = self.train.shape[
            0], self.train.shape[1], self.train.shape[2:]
        self.num_test_samples =  self.test.shape[1]
        if flatten:
            self.train = self.train.reshape([self.num_frames, self.num_train_samples, -1])
            self.test = self.test.reshape([self.num_frames, self.num_test_samples, -1])
        self.train_index = 0
        self.testing_index = 0
        
        print("loading of UCSD ped1 completed")
        print(self.train.shape)
    
    def shuffle(self):
        indices = np.random.permutation(self.num_train_samples)
        self.train = self.train[:, indices, ...]
    
    def get_batch(self, set, batch_size):
        if set == "train":
            if self.train_index + batch_size - 1 >= self.num_train_samples:
                self.shuffle()
                self.train_index = 0

            batch = self.train[:,
                               self.train_index:self.train_index + batch_size, ...]
            self.train_index += batch_size
            return batch
        elif set == "anomaly":
            if self.testing_index + batch_size - 1 >= self.test.shape[1]:
                self.testing_index = 0
            batch = self.test[:,
                              self.testing_index: self.testing_index + batch_size, ...]
            l = self.labels[:, self.testing_index: self.testing_index + batch_size]
            self.testing_index += batch_size
            return batch, l
    

if __name__ =="__main__":
    
    dl = UCSD_loader("UCSD_Anomaly_Dataset.v1p2/UCSD_Anomaly_")
    data, l  = dl.get_batch("anomaly", 35)
    for i in range(100):
            x1 = data[i][0].reshape(64, 64)
            plt.figure(1)
            plt.axis("off")
            plt.clf()
            plt.title('output')
            plt.imshow(x1, cmap="gray")
            plt.draw()
            plt.pause(0.5)
            #print(i)
            print(l[i][1])
'''    
    path = 'movingMnist/mnist_test_seq_16.npy'
    time_steps = 20
    data_loader = Moving_MNIST_Loader(path=path, time_steps=time_steps, flatten=True)
    d, l = data_loader.get_batch("", 10)
    sum = 0
    for i in range(int(1000/32)):
        d, l = data_loader.get_batch("", 32)
        print(len(l))
    
    for i in range(20):
        x1 = data_loader.test[i][0].reshape(16, 16)
        plt.figure(1)
        plt.axis("off")
        plt.clf()
        plt.title('output')
        plt.imshow(x1, cmap="gray")
        plt.draw()
        plt.pause(1)
        print(data_loader.labels[i][0])
        plt.savefig("corrupted_frame"+str(i)+".png")

'''