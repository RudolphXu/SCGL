import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import FinalRowWise_layer, SingularValue_layer, Weighted_layer
import numpy as np
import matplotlib.pyplot as plt
## granger full rank
class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
#        print(self.cuda)
        self.m = data.m
        self.w = args.window

        self.batch_size = args.batch_size
        self.k_list = args.k_list
        self.add_k_list = [self.m]+self.k_list
        self.len_k_list = len(self.k_list)
        self.pre_win = args.pre_win; 
        self.p_list = args.p_list
        self.p_allsum = np.sum(self.p_list)
        self.len_p_list = len(self.p_list)
        self.compress_p_list = args.compress_p_list
        self.len_compress_p_list = len(self.compress_p_list)

        self.sparse_label = []
        self.orthgonal_label = []
        ## p_list extention
        self.P = []; #w->hid
        for p_i in np.arange(0,self.len_p_list):
            self.P.append(Weighted_layer.WL_Model())
        self.P = nn.ModuleList(self.P);
        
        self.linears = [(nn.Linear(self.w, self.p_list[0]))]; #w->hid
        self.sparse_label.append(0); self.orthgonal_label.append(1);
        #self.linears.append(  nn.InstanceNorm1d(self.m)); #m->k
        #self.sparse_label.append(0); self.orthgonal_label.append(0);
        if self.len_p_list>1:
            for p_i in np.arange(1,self.len_p_list):
                self.linears.append((nn.Linear(self.p_list[p_i-1], self.p_list[p_i]))); #w->hid
                self.sparse_label.append(0); self.orthgonal_label.append(1);

        
        ## graph layers
        for p_i in np.arange(0,self.len_p_list):
            for k_i in np.arange(0,self.len_k_list):
                self.linears.append( nn.utils.weight_norm(nn.Linear(self.add_k_list[k_i], self.add_k_list[k_i+1], bias = False))); #m->k
                self.sparse_label.append(1); 
                if k_i ==0:
                    self.orthgonal_label.append(1)
                else:
                    self.orthgonal_label.append(2)
                self.linears.append(nn.BatchNorm1d(self.p_list[-1])); #m->k
                self.sparse_label.append(0); self.orthgonal_label.append(0);

            
            self.linears.append(SingularValue_layer.SV_Model(data, self.k_list[-1], self.use_cuda)) #k->k
            self.sparse_label.append(0); self.orthgonal_label.append(0);
            
            for k_i in np.arange(self.len_k_list,0,-1):       
                self.linears.append( nn.utils.weight_norm(nn.Linear(self.add_k_list[k_i], self.add_k_list[k_i-1], bias = False))); #m->m, supervised
                self.sparse_label.append(1); 
                if k_i == 1:
                    self.orthgonal_label.append(1)
                else:
                    self.orthgonal_label.append(2)
                self.linears.append(nn.BatchNorm1d(self.p_list[-1])); #m->k
                self.sparse_label.append(0); self.orthgonal_label.append(0);
                
                
        #self.linears.append(  nn.BatchNorm1d(self.m)); #m->k
        #self.sparse_label.append(0); self.orthgonal_label.append(0)
                
        if self.len_compress_p_list>0:
            self.linears.append( (nn.Linear(self.p_allsum, self.compress_p_list[0])))
            self.sparse_label.append(0); self.orthgonal_label.append(1);
            for p_j in np.arange(1,self.len_compress_p_list):
                self.linears.append( (nn.Linear(self.compress_p_list[p_j-1], self.compress_p_list[p_j])))
                self.sparse_label.append(0); self.orthgonal_label.append(1);
          
#        
        self.linears.append(FinalRowWise_layer.FR_Model(args, data)); #k->k  
        self.sparse_label.append(1); self.orthgonal_label.append(0);
        
        
        
        self.linears = nn.ModuleList(self.linears);
        self.dropout = nn.Dropout(args.dropout);

        for layer_i in range(len(self.linears)):
            if not isinstance(self.linears[layer_i], nn.InstanceNorm1d) and not isinstance(self.linears[layer_i], nn.BatchNorm1d) and not isinstance(self.linears[layer_i], SingularValue_layer.SV_Model):
                W = self.linears[layer_i].weight.transpose(0,1).detach().numpy()
                ## sparsity
                if W.ndim >=2 and self.orthgonal_label[layer_i]==1: ## sparsity
                    #nn.init.xavier_normal_(self.linears[layer_i].weight)
                    self.linears[layer_i].weight = nn.init.orthogonal_(self.linears[layer_i].weight)
                if W.ndim >=2 and self.orthgonal_label[layer_i]>1: ## sparsity
                    #nn.init.xavier_normal_(self.linears[layer_i].weight)
                    tmp = self.linears[layer_i].weight
                    self.linears[layer_i].weight = np.eye(tmp.shape[0],tmp.shape[1])
                    
    
    def forward(self, inputs):
        x_input = inputs[0] #pxm 
        x = x_input.transpose(2,1).contiguous(); #mxp
        x = self.dropout(x)            
        x_org = x
        x_p = []

        if self.p_list[0]> self.w:
            padding = nn.ConstantPad2d((0, self.p_list[0]-self.w, 0, 0), 0)
            x_0n = padding(x_org)
        
        x_0 = x_org
        for layer_i in range(self.len_p_list):  
            #pl = 1-0.5*layer_i/self.len_p_list
            x_i = self.linears[layer_i](x_0);
            #x_i = F.relu(x_i)
            x_i = F.relu(self.P[layer_i](x_i) + x_0n)
            x_0n = x_i
            x_0 = x_i
            x_p.append(x_i)
        
        x_p_m = []  
        for layer_i in range(self.len_p_list):
            
            x_sp =  x_p[layer_i].transpose(2,1).contiguous(); ## read the data piece  
            
            x_sp_tmp = []
            x_sp_tmp.append(x_sp)
            for k_i in np.arange(0,self.len_k_list):   
                x_sp = self.linears[self.len_p_list+layer_i*(4*self.len_k_list+1)+2*k_i](x_sp);  #lxk 
                x_sp = self.linears[self.len_p_list+layer_i*(4*self.len_k_list+1)+2*k_i+1](x_sp);  #lxk 
                x_sp = F.tanh(x_sp/5.);
                x_sp = self.dropout(x_sp)
                x_sp_tmp.append(x_sp)
                #x_sp = self.dropout(x_sp)
            
            #pdb.set_trace()
            x_sp = self.linears[self.len_p_list+layer_i*(4*self.len_k_list+1) + 2*self.len_k_list](x_sp);  #lxk 
            #x_sp = F.layer_norm(x_sp, [x_sp.shape[-2],x_sp.shape[-1]])
            
            for k_i in np.arange(0,self.len_k_list):  
                #pdb.set_trace()
                #if k_i>0:
                #    x_sp = x_sp + x_sp_tmp[-1*(k_i+1)]
                x_sp = self.linears[self.len_p_list+layer_i*(4*self.len_k_list+1)+2*self.len_k_list + 1+2*k_i](x_sp);  #lxm
                x_sp = self.linears[self.len_p_list+layer_i*(4*self.len_k_list+1)+2*self.len_k_list + 1+2*k_i+1](x_sp);  #lxm
                #x_sp = x_sp + x_sp_tmp[-1*(k_i+2)]
                x_sp = F.relu(x_sp/1.);
                x_sp = self.dropout(x_sp)
            
            x_sp = x_sp.transpose(2,1).contiguous(); #mxl
            x_p_m.append(x_sp)
            
        x_p_m = torch.cat(x_p_m, dim = 2) 
        #x_p_m = self.linears[self.len_p_list+self.len_p_list*(2*self.len_k_list+1)](x_p_m);
        
            
        if self.len_compress_p_list>0:
            for p_j in range(self.len_compress_p_list): 
                x_p_m = self.linears[self.len_p_list+self.len_p_list*(4*self.len_k_list+1)+p_j](x_p_m); #mx2
                x_p_m = F.tanh(x_p_m/5.);
                x_sp = self.dropout(x_sp)
         
        #pdb.set_trace()
        
        final_y = self.linears[-1](x_p_m)

        return final_y      
     
    def predict_relationship(self):
        CGraph_list = []
        G = np.zeros((self.m,self.m))
                       
        for layer_i in range(self.len_p_list):
            pl = self.P[layer_i].weight.data
            
            A = self.linears[self.len_p_list+layer_i*(4*self.len_k_list+1)+0].weight.transpose(0,1).cpu().detach().numpy()
            B = np.diag(self.linears[self.len_p_list+layer_i*(4*self.len_k_list+1)+2*self.len_k_list].weight.transpose(0,1).detach().cpu().numpy().ravel())
            C = self.linears[self.len_p_list+layer_i*(4*self.len_k_list+1)+4*self.len_k_list+1-2].weight.transpose(0,1).cpu().detach().numpy()
           
            CGraph = np.abs(np.dot(np.dot(A,B),C))
            CGraph[range(self.m), range(self.m)] = 0    
            CGraph_list.append(CGraph)
            G = np.add(G, np.multiply(CGraph, pl.cpu().detach().numpy())) 
      
        G[range(self.m), range(self.m)] = 0 
                  
        return G
    
    def predict_relationship2(self):
        CGraph_list = []
        G = np.zeros((self.m,self.m))
                       
        for layer_i in range(self.len_p_list):
            pl = self.P[layer_i].weight.data
            
            A = self.linears[self.len_p_list+layer_i*(4*self.len_k_list+1)+0].weight.transpose(0,1).cpu().detach().numpy()
            B = np.diag(self.linears[self.len_p_list+layer_i*(4*self.len_k_list+1)+2*self.len_k_list].weight.transpose(0,1).detach().cpu().numpy().ravel())
            C = self.linears[self.len_p_list+layer_i*(4*self.len_k_list+1)+4*self.len_k_list+1-2].weight.transpose(0,1).cpu().detach().numpy()
           
            CGraph = np.abs(np.dot(np.dot(A,B),C))
#            CGraph[range(self.m), range(self.m)] = 0    
            CGraph_list.append(CGraph)
            G = np.add(G, np.multiply(CGraph, pl.cpu().detach().numpy())) 
      
 #       G[range(self.m), range(self.m)] = 0 
                  
        return G