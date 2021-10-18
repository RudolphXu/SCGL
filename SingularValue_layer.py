import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from sklearn.preprocessing import normalize
## granger full rank

class SV_Model(nn.Module):
    def __init__(self, data, lowrank, cuda):
        self.cuda = cuda
        super(SV_Model, self).__init__()
        self.weight = nn.Parameter(torch.ones([lowrank,1]))
        #pdb.set_trace()
        #self.weight.data = torch.nn.Parameter(torch.from_numpy(data.GTs).float()).reshape(lowrank,1);
        
        
    def forward(self, x):
        #l = x.shape[1]
        k = x.shape[2]
        
        y = torch.Tensor(x.shape)
        

        for j in range(k):
            
            tmp_new = torch.mul(x[:,:,j], self.weight[j,0])
            y[:,:,j] = tmp_new
            
        #pdb.set_trace()
        
        if self.cuda:
            y = y.cuda()
        
        
        return y