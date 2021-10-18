import torch
import numpy as np;
from torch.autograd import Variable
from scipy.io import loadmat

class Data_utility(object):
    def __init__(self, args):
        self.cuda = args.cuda
        self.model = args.model
        self.P = args.window
        self.h = args.horizon
        self.random_shuffle = args.random_shuffle
        
        self.pre_win = args.pre_win 

        self.rawdat = loadmat(args.data)['expression']
        self.graph = loadmat(args.graph_path)['A']
        print('data shape', self.rawdat.shape)

        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self._normalized()
        
        self._split(int(args.train*self.n), int((args.train+args.valid)*self.n), self.n)
        
    def _normalized(self):
        
        for i in range(self.m):
            Mean = np.mean(self.rawdat[:,i])
            Std = np.std(self.rawdat[:,i])
            self.dat[:,i] = (self.rawdat[:,i] - Mean)/Std

    def _split(self, train, valid, test):

        train_set = range(self.P+self.h-1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)

        self.train0 = self._batchify_DCM(train_set, self.h);
        self.valid0 = self._batchify_DCM(valid_set, self.h);
        self.test = self._batchify_DCM(test_set, self.h);
        
        if self.random_shuffle:
            self.X_all = torch.cat((self.train0[0], self.valid0[0]), dim = 0)
            self.Y_all = torch.cat((self.train0[1], self.valid0[1]), dim = 0)
        
        numTrain = self.train0[0].shape[0]
        numValid = self.valid0[0].shape[0]
         
        random_index = torch.randperm(numTrain + numValid)

        self.X_all = self.X_all[random_index, :, :]
        self.Y_all = self.Y_all[random_index, :, :]
        
    
        self.train = [self.X_all[:numTrain,:,:], self.Y_all[:numTrain,:,:]]
        self.valid = [self.X_all[numTrain:,:,:], self.Y_all[numTrain:,:,:]]
        

    def _batchify_DCM(self, idx_set, horizon):

        n = len(idx_set)

        X = torch.zeros((n, self.P, self.m))
        if self.pre_win == 1:
            Y = torch.zeros((n, self.m));
        else:        
            Y = torch.zeros((n, self.pre_win, self.m))
        
        for i in range(n-self.pre_win+1):
            end = idx_set[i];
            start = end - self.P;
            
            norm_x = self.dat[start:end, :]

            X[i,:self.P,:] = torch.from_numpy(norm_x);
            
            if self.pre_win ==1:
                norm_y = self.dat[idx_set[i], :]
                Y[i,:] = torch.from_numpy(norm_y);
            else:    
                norm_y = self.dat[idx_set[i]:idx_set[i]+self.pre_win, :]

                Y[i,:,:] = torch.from_numpy(norm_y);
        #pdb.set_trace()
        index = torch.randperm(len(X))
        return [X[index], Y[index]]

    def get_batches(self, data, batch_size, shuffle = False):
        
        inputs = data[0]
        targets = data[1]
        #pdb.set_trace()
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]; 
            Y = targets[excerpt];
            
            if (self.cuda):
                X = X.cuda();
                Y = Y.cuda();                

            data = [[Variable(X)], Variable(Y)]
            yield data;
            start_idx += batch_size
