#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset
import torch.optim as optim
import torchvision.models as models
import numpy as np
import os,sys,os.path
from tensorboardX import SummaryWriter
import pickle
from tqdm import tqdm
import copy
import gc
import time
import torch.nn.functional as F


# In[5]:


from option import args_parser
from utils import Accuracy,average_weights
from sampling import LocalDataset, LocalDataloaders , partition_data
from resnet_fedalign import resnet18 as resnet18_fedalign


# In[6]:


torch.set_default_dtype(torch.float64)
print(torch.__version__)
torch.cuda.is_available()
device = torch.device("cuda:0")
print(device)
args = args_parser()
np.set_printoptions(threshold=np.inf)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'


# In[8]:


args = args_parser()
args.num_clients = 10
args.code_len = 64
args.batch_size = 32

# In[9]:


global_model = resnet18_fedalign(10)
print('# model parameters:', sum(param.numel() for param in global_model.parameters()))
global_model = nn.DataParallel(global_model)
global_model.to(device)


# In[10]:


train_dataset,testset, dict_users, dict_users_test = partition_data(n_users = args.num_clients, alpha=0.3,rand_seed = 0, dataset='CIFAR10')


# In[11]:


Loaders_train = LocalDataloaders(train_dataset,dict_users,args.batch_size,ShuffleorNot = True,frac=0.1)
Major_classes = []
for idx in range(args.num_clients):
    counts = [0]*10
    for batch_idx,(X,y) in enumerate(Loaders_train[idx]):
        batch = len(y)
        y = np.array(y)
        for i in range(batch):
            counts[int(y[i])] += 1
    print(counts)


# In[12]:


Loaders_test = LocalDataloaders(testset,dict_users_test,args.batch_size,ShuffleorNot = True,frac=0.2)
Major_classes = []
for idx in range(args.num_clients):
    counts = [0]*10
    for batch_idx,(X,y) in enumerate(Loaders_test[idx]):
        batch = len(y)
        y = np.array(y)
        for i in range(batch):
            counts[int(y[i])] += 1
    print(counts)


# In[13]:


logger = SummaryWriter('./logs')
checkpoint_dir = './checkpoint/'+ args.dataset + '/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
with open(checkpoint_dir+'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)
print('Data and model loaded')
print('Checkpoint dir:', checkpoint_dir)


# In[14]:


for m in global_model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))


# In[15]:


def soft_predict(Z,temp):
    m,n = Z.shape
    Q = torch.zeros(m,n)
    Z_sum = torch.sum(torch.exp(Z/temp),dim=1)
    for i in range(n):
        Q[:,i] = torch.exp(Z[:,i]/temp)/Z_sum
    return Q


# In[16]:


class LocalUpdate(object):
    """
    This class is for train the local model with input global model(copied) and output the updated weight
    args: argument 
    Loader_train,Loader_val,Loaders_test: input for training and inference
    user: the index of local model
    idxs: the index for data of this local model
    logger: log the loss and the process
    """
    def __init__(self, args, Loader_train,loader_test,idxs, logger, code_length, num_classes, device):
        self.args = args
        self.logger = logger
        self.trainloader = Loader_train
        self.testloader = loader_test
        self.idxs = idxs
        self.ce = nn.CrossEntropyLoss() 
        self.device = device
        self.code_length = code_length
        self.kld = nn.KLDivLoss()
        self.mse = nn.MSELoss()
        self.model = copy.deepcopy(global_model)
#         self.model = nn.DataParallel(self.model).to(device)
        self.width_range = [0.25,1]
        self.mu = 0.45
        
    def update_weights_align(self,global_round):
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = optim.Adam(self.model.parameters(),lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                self.model.module.apply(lambda m: setattr(m, 'width_mult', self.width_range[-1]))
                t_feats, p = self.model.module.extract_feature(X)
                loss = self.ce(p,y)    
                loss.backward()
                loss_CE = loss.item()
                self.model.module.apply(lambda m: setattr(m, 'width_mult', self.width_range[0]))
                s_feats = self.model.module.reuse_feature(t_feats[-2].detach())
                
                # Lipschitz loss
                TM_s = torch.bmm(self.transmitting_matrix(s_feats[-2], s_feats[-1]), self.transmitting_matrix(s_feats[-2], s_feats[-1]).transpose(2,1))
                TM_t = torch.bmm(self.transmitting_matrix(t_feats[-2].detach(), t_feats[-1].detach()), self.transmitting_matrix(t_feats[-2].detach(), t_feats[-1].detach()).transpose(2,1))
                loss = F.mse_loss(self.top_eigenvalue(K=TM_s), self.top_eigenvalue(K=TM_t))
                loss = self.mu *(loss_CE/loss.item())*loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
                
                loss.backward()
                
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(X),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return self.model.state_dict(),sum(epoch_loss) / len(epoch_loss)
   
    def transmitting_matrix(self, fm1, fm2):
        if fm1.size(2) > fm2.size(2):
            fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))

        fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
        fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1, 2)

        fsp = torch.bmm(fm1, fm2) / fm1.size(2)
        return fsp

    def top_eigenvalue(self, K, n_power_iterations=10, dim=1):
        v = torch.ones(K.shape[0], K.shape[1], 1).to(self.device)
        for _ in range(n_power_iterations):
            m = torch.bmm(K, v)
            n = torch.norm(m, dim=1).unsqueeze(1)
            v = m / n

        top_eigenvalue = torch.sqrt(n / torch.norm(v, dim=1).unsqueeze(1))
        return top_eigenvalue
        
    def test_accuracy(self):
        self.model.eval()
        self.model.apply(lambda m: setattr(m, 'width_mult', self.width_range[-1]))
        accuracy = 0
        cnt = 0
        for batch_idx, (X, y) in enumerate(self.testloader):
            X = X.to(self.device)
            y = y.to(self.device)
            p = self.model(X).double()
            y_pred = p.argmax(1)
            accuracy += Accuracy(y,y_pred)
            cnt += 1
        return accuracy/cnt
    
    
    def load_model(self,global_weights):
        self.model.load_state_dict(global_weights)


# In[17]:


global_weights = global_model.state_dict()


# In[18]:


# training
args.num_epochs = 50
train_loss, train_accuracy = [], []
val_acc_list, net_list = [], []
cv_loss, cv_acc = [], []
print_every = 2
val_loss_pre, counter = 0, 0
LocalModels = []
loader_test = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=2)
for idx in range(args.num_clients):
    LocalModels.append(LocalUpdate(args,Loaders_train[idx], Loaders_test[idx], idxs=dict_users[idx], 
                                   logger=logger, code_length = args.code_len, num_classes = 10, device=device))
    


# In[ ]:


for epoch in tqdm(range(args.num_epochs)):
    test_accuracy = 0
    begin_time = time.time()
    Knowledges = []
    local_weights, local_losses = [], []
    print(f'\n | Global Training Round : {epoch+1} |\n')
    global_model.train()
    m = max(int(args.sampling_rate * args.num_clients), 1)
    idxs_users = np.random.choice(range(args.num_clients), m, replace=False)
    for idx in idxs_users:
        LocalModels[idx].load_model(global_weights)
        w, loss = LocalModels[idx].update_weights_align(global_round=epoch)
        local_losses.append(copy.deepcopy(loss))
        local_weights.append(copy.deepcopy(w))
        acc = LocalModels[idx].test_accuracy()
        test_accuracy += acc
    
            
     # update global weights
    global_weights = average_weights(local_weights)
                
  

    loss_avg = sum(local_losses) / len(local_losses)
    train_loss.append(loss_avg)
    print("average loss:  ", loss_avg)
    print('average test accuracy:', test_accuracy / args.num_clients)

    end_time = time.time()
    training_time  = end_time - begin_time
    print('training time: ', training_time)


# In[ ]:


global_model.load_state_dict(global_weights)
loader_test = torch.utils.data.DataLoader(testset, batch_size=64,shuffle=False, num_workers=2)
accuracy = 0
cnt = 0
global_model.apply(lambda m: setattr(m, 'width_mult', 1))
global_model.eval()
for cnt, (X,y) in enumerate(loader_test):
    X = X.to(device)
    y = y.double().to(device)
    p = global_model(X)
    y_pred = p.argmax(1).double()
    accuracy += Accuracy(y,y_pred)
    cnt += 1
print("accuracy of test:",accuracy/cnt)

