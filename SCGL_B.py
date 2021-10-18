
# coding: utf-8

# In[1]:


import math
import time

import torch
import torch.nn as nn
import DCM_DeepCausal
import numpy as np;
#import importlib
from utils import *;

import Optim
import scipy
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.sparse.linalg import svds
from numpy import linalg as LA


# In[2]:


#default setting

class args:
    data = 'data/syntheticB/filter_norm_expression0.mat'
    graph_path = 'data/syntheticB/groundtruth.mat'
    train = 0.80
    valid = 0.19
    model = 'DCM_DeepCausal'
    
    window = 5
    k_list = [50]#[30,25,20]
    #lowrank = 20

    random_shuffle = True
    
    pre_win = 3
    p_list = [20]*6#[20,20,20,20]#[20,20,20,20,20]
    compress_p_list = [50]#[:q40,40]
    L1Loss = False#False
    clip = 1.
    epochs = 150
    batch_size = 32
    dropout = 0.1
    seed = 12345
    gpu = 0
    save = 'save/model.pt'
    cuda = False
    optim = 'adam'#'adam'
    lr = 0.01
    weight_decay = 0
    horizon = 1


# In[3]:


rawdat = sio.loadmat(args.data)['expression']
plt.plot(rawdat)


# In[4]:


rawdat.shape


# In[5]:


manualSeed = args.seed

np.random.seed(manualSeed)
#random.seed(manualSeed)
torch.manual_seed(manualSeed)
if args.cuda is True:
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# In[6]:


if args.cuda:
    print("a")
    torch.cuda.set_device(args.gpu)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    print("b")
    if not args.cuda:
        print("c")
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        print("d")
        torch.cuda.manual_seed(args.seed)


# In[7]:


Data = Data_utility(args)

GroundTruth = sio.loadmat(args.graph_path)['A']
GroundTruth_flat = GroundTruth.reshape(Data.m*Data.m)
# In[ ]:


print('buliding model')
model = eval(args.model).Model(args, Data);


print(model.linears)

if args.cuda:
    model.cuda()

nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

if args.L1Loss:
    criterion = nn.L1Loss(size_average=False);
else:
    criterion = nn.MSELoss(size_average=False);#,reduce=False);
evaluateL2 = nn.MSELoss(size_average=False);
evaluateL1 = nn.L1Loss(size_average=False)
if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda();
    evaluateL2 = evaluateL2.cuda();


best_val = 10000000;
optim = Optim.Optim(
    model.parameters(), args.optim, args.lr, args.clip, weight_decay = args.weight_decay,
)
test_acc, test_rae, test_corr = 0, 0, 0


# In[ ]:


def evaluate(loader, data, model, evaluateL2, evaluateL1, batch_size):
    model.eval();
    total_loss = 0;
    n_samples = 0;


    for inputs in loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]

        output = model(X);

        loss_org = criterion(output, Y);
        total_loss += loss_org.data.item();
          
        n_samples += (output.size(0) * loader.m);
    
    return total_loss / n_samples


# In[ ]:


def train(loader, data, model, criterion, optim, batch_size, GroundTruth_flat):
    model.train();
    total_loss = 0;
    mse = 0
    n_samples = 0;
    

    
    total_time = 0
    
    for inputs in loader.get_batches(data, batch_size, True):        
        begin_time1 = time.time()
        X, Y = inputs[0], inputs[1]

        model.zero_grad();
        output = model(X);
        loss_org = criterion(output, Y);
        
        end_time1 = time.time()
        dt1 = end_time1 - begin_time1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    
        
        l2_reg = None
        otg_reg = None

        for layer_i in range(len(model.linears)):
            if not isinstance(model.linears[layer_i], nn.InstanceNorm1d) and not isinstance(model.linears[layer_i], nn.BatchNorm1d):
                W = model.linears[layer_i].weight.transpose(0,1).cpu().detach().numpy()
                ## sparsity
                if W.ndim >=2 and W.shape[0]*W.shape[1]>=100 and model.sparse_label[layer_i]>0: ## sparsity
                    if l2_reg is None:
                        l2_reg = LA.norm(W)
                    else:
                        l2_reg = l2_reg + LA.norm(W)
                ## orthgonality
                if W.ndim >=2 and model.orthgonal_label[layer_i]==1: 
                    if otg_reg is None:
                        otg_reg = LA.norm(np.abs(np.dot(W,np.transpose(W))-np.eye(W.shape[0])))
                    else:
                        otg_reg = otg_reg + LA.norm(np.abs(np.dot(W, np.transpose(W))-np.eye(W.shape[0])))
        
        batch_loss = loss_org + abs(l2_reg)*1.0+ abs(otg_reg)*0.1 

        
        begin_time2 = time.time()
        
        batch_loss.backward()
        total_loss += batch_loss.data.item();
        mse += loss_org.data.item()
        
        end_time2 = time.time()
        
        dt2 = end_time2 - begin_time2
        
        
        total_time = total_time + dt1 + dt2
        
                
        optim.step();
        n_samples += (output.size(0) * loader.m);
            
    return total_loss / n_samples, mse / n_samples, total_time


# In[ ]:


print("begin training")
train_loss_set = []
test_loss_set = []
mse_set = []
L1_W_loss_set = []
L1_L_loss_set = []
AUC_1 = []
AUC_2 = []
AUC_3 = []
AUC_4 = []

num_weight_plot = min([len(model.linears),50])
weight_norm = np.zeros((args.epochs, num_weight_plot))
auc1_best = 0.1
G_best = np.zeros((model.m, model.m))
#pdb.set_trace()
weight_matrix = []
for epoch in range(0, args.epochs):
    #if epoch == 100:
    #    pdb.set_trace()
    P_weight=np.zeros(len(args.p_list))
    for j in range(len(args.p_list)):
        P_weight[j] = model.P[j].weight
    weight_matrix.append(P_weight)

    
    train_loss, mse, epoch_time = train(Data, Data.train, model, criterion, optim, args.batch_size, GroundTruth_flat)
    val_loss = evaluate(Data, Data.valid, model, evaluateL2, evaluateL1, args.batch_size);
    epoch_end_time = time.time()
    
    CGraph1, CGraph2, CGraph3, CGraph4 = model.predict_relationship()
    CGraph = CGraph1.reshape(model.m*model.m)
    #fpr, tpr, thresholds = metrics.roc_curve(GroundTruth_flat, CGraph)
    #auc1 = (metrics.auc(fpr, tpr))
    precision, recall, threshold = metrics.precision_recall_curve(GroundTruth_flat, CGraph)
    auc1 = metrics.auc(recall, precision)
    CGraph = CGraph2.reshape(model.m*model.m)
    #fpr, tpr, thresholds = metrics.roc_curve(GroundTruth_flat, CGraph)
    #auc2 = (metrics.auc(fpr, tpr))
    precision, recall, threshold = metrics.precision_recall_curve(GroundTruth_flat, CGraph)
    auc2 = metrics.auc(recall, precision)
    CGraph = CGraph1.reshape(model.m*model.m)
    fpr, tpr, thresholds = metrics.roc_curve(GroundTruth_flat, CGraph)
    auc3 = (metrics.auc(fpr, tpr))
    #precision, recall, threshold = metrics.precision_recall_curve(GroundTruth_flat, CGraph)
    #auc3 = metrics.auc(recall, precision)
    CGraph = CGraph2.reshape(model.m*model.m)
    fpr, tpr, thresholds = metrics.roc_curve(GroundTruth_flat, CGraph)
    auc4 = (metrics.auc(fpr, tpr))
    #precision, recall, threshold = metrics.precision_recall_curve(GroundTruth_flat, CGraph)
    #auc4 = metrics.auc(recall, precision)
    
    for layer_i in range(num_weight_plot):
        if not isinstance(model.linears[layer_i], nn.InstanceNorm1d):
            tmp = model.linears[layer_i].weight.cpu().detach().numpy()
            if tmp.ndim >2:
                tmp = tmp[:,:,0]
            weight_norm[epoch, layer_i] = LA.norm(tmp)
        
    
    train_loss_set.append(train_loss)
    test_loss_set.append(val_loss)
    mse_set.append(mse)
    AUC_1.append(auc1)
    AUC_2.append(auc2)
    AUC_3.append(auc3)
    AUC_4.append(auc4)
    print('|end_epoch{:3d}|time:{:5.2f}s|tn_ls {:5.8f}| mse {:5.8f} |vd_ls {:5.4f}|auc1 {:5.4f}|auc2 {:5.4f}|auc3 {:5.4f}|auc4 {:5.4f}'.format(epoch, epoch_time, train_loss, mse, val_loss, auc1, auc2, auc3, auc4))
    # Save the model if the validation loss is the best we've seen so far.
    
    if auc1>=auc1_best:
        G_best = CGraph1
               
    if val_loss < best_val:
        best_val = val_loss
        test_loss  = evaluate(Data, Data.test, model, evaluateL2, evaluateL1, args.batch_size);
        


# In[ ]:


#pdb.set_trace() 
plt.figure(figsize=(15,3.5))
plt.subplot(141)
plt.imshow(GroundTruth)
plt.subplot(142)
plt.imshow(G_best)
plt.subplot(143)
plt.imshow(CGraph1)
plt.subplot(144)
plt.imshow(CGraph2)



