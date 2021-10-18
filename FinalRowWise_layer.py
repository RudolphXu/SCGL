import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from sklearn.preprocessing import normalize
import math
## granger full rank

class FR_Model(nn.Module):
    def __init__(self, args, data):
        super(FR_Model, self).__init__()
        self.pre_win = args.pre_win
        self.m = data.m
        self.p_list = (args.p_list) 
        self.len_p_list = len(args.p_list) 
        self.compress_p_list = args.compress_p_list
        self.p_allsum = np.sum(self.p_list)
        self.len_compress_p_list = len(self.compress_p_list)
        self.cuda = args.cuda
        if self.len_compress_p_list>0:
            
            self.compress_p = args.compress_p_list[-1]
            self.weight = nn.Parameter(torch.ones([self.m, self.compress_p, self.pre_win]))
        else:
            self.weight = nn.Parameter(torch.ones([self.m, self.p_allsum, self.pre_win]))
        
        #nn.init.orthogonal_(self.weight)
        #nn.init.sparse_(self.weight, sparsity=0.3)
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.orthogonal_(self.weight)
        self.bias = Parameter(torch.Tensor(self.m,self.pre_win)) 
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        #self.weight = nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
        
        
        
    def forward(self, x):
        #l = x.shape[1]
        #k = x.shape[2]
        
        if self.pre_win ==1:
            final_y = torch.empty(x.shape[0], self.m) 
        else :
            final_y = torch.empty(x.shape[0], self.pre_win, self.m) 
        
        #x_latest = x_latest.view(x_latest.shape[0], x_latest.shape[1], 1)
        #x = torch.cat((x, x_latest), 2)
        for j in range(self.m):           
            if self.pre_win ==1:   
                #pdb.set_trace()
                final_y[:,j] = F.linear(x[:,j,:], self.weight[j,:].view(1, self.weight.shape[1]), self.bias[j,:]).view(-1);               
            else:
                #pdb.set_trace()
                final_y[:,:,j] = F.linear(x[:,j,:], self.weight[j,:].transpose(1,0), self.bias[j,:]);               
        #pdb.set_trace()
        
        if self.cuda:
            final_y = final_y.cuda()
        
        return final_y;
    
    def get_pi_weight(self):
        if self.len_compress_p_list>0:
            func_1 = nn.MaxPool1d(kernel_size=self.compress_p, stride=self.compress_p)
        else:
            func_1 = nn.MaxPool1d(kernel_size=self.p_list[0], stride=self.p_list[0])
        func_2 = nn.MaxPool1d(kernel_size=self.m, stride=self.m)
        
        weight1_norm_all = np.zeros((self.weight.shape[0], self.len_p_list))
        weight2_norm_all = np.zeros((self.len_p_list))
        for layer_i in range(self.weight.shape[-1]):
            weight_tmp = self.weight[:,:,layer_i]
            weight0 = weight_tmp.view(1, self.weight.shape[0],self.weight.shape[1])
            weight1 = func_1(torch.abs(weight0)) 
            weight1_inv = weight1.transpose(2,1).contiguous(); #mxp
            weight2 = func_2(weight1_inv).detach().numpy().ravel() 
            weight2_norm = weight2/np.sum(weight2)
            weight1_norm = F.normalize(weight1, p=1, dim=1).view(weight1.shape[1], weight1.shape[2]).detach().numpy()
            
            weight1_norm_all = weight1_norm_all + weight1_norm
            #pdb.set_trace()
            weight2_norm_all = weight2_norm_all + weight2_norm
        
        #pdb.set_trace()
        return weight1_norm_all, weight2_norm_all
        
        
    
#    m = SingularValue_layer(5)
#    x = torch.randn(30, 10, 5)
#    output = m(x)
    
    

