#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import torch.nn.functional as F


# In[2]:


from option import args_parser
from utils import Accuracy,average_weights
from sampling import LocalDataset, LocalDataloaders , partition_data


# In[3]:


torch.set_default_dtype(torch.float64)
print(torch.__version__)
torch.cuda.is_available()
device = torch.device("cuda:0")
print(device)
args = args_parser()
np.set_printoptions(threshold=np.inf)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'


# In[4]:


args = args_parser()
args.num_clients = 10
args.code_len = 64
args.batch_size = 32


# In[5]:


class net(nn.Module):
    def __init__(self,
                 code_length=64, 
                 num_classes = 10,
                 ):
        super(net,self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes  
        self.feature_extractor = models.resnet18(num_classes=self.code_length)
        self.classifier =  nn.Sequential(
                                nn.Linear(self.code_length, self.num_classes))
    def forward(self,x): #x = [batch,time,freq]
        f = self.feature_extractor(x)
        z = self.classifier(f)
        return z
    
global_model = net(code_length=64, num_classes = 10)
print('# model parameters:', sum(param.numel() for param in global_model.parameters()))
global_model = nn.DataParallel(global_model)
global_model.to(device)


# In[6]:


class Generator(nn.Module):
    def __init__(self, embedding=False, latent_layer_idx=-1):
        super(Generator, self).__init__()
        self.embedding = embedding
        self.latent_layer_idx = latent_layer_idx
        self.hidden_dim, self.latent_dim, self.input_channel, self.n_class, self.noise_dim = (512, 64, 3, 10, 64)
        input_dim = self.noise_dim * 2 if self.embedding else self.noise_dim + self.n_class
        self.fc_configs = [input_dim, self.hidden_dim]
        self.init_loss_fn()
        self.build_network()

    def get_number_of_parameters(self):
        pytorch_total_params=sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params

    def init_loss_fn(self):
        self.crossentropy_loss=nn.NLLLoss(reduce=False) # same as above
        self.diversity_loss = DiversityLoss(metric='l1')
        self.dist_loss = nn.MSELoss()

    def build_network(self):
        if self.embedding:
            self.embedding_layer = nn.Embedding(self.n_class, self.noise_dim)
        ### FC modules ####
        self.fc_layers = nn.ModuleList()
        for i in range(len(self.fc_configs) - 1):
            input_dim, out_dim = self.fc_configs[i], self.fc_configs[i + 1]
            print("Build layer {} X {}".format(input_dim, out_dim))
            fc = nn.Linear(input_dim, out_dim)
            bn = nn.BatchNorm1d(out_dim)
            act = nn.ReLU()
            self.fc_layers += [fc, bn, act]
        ### Representation layer
        self.representation_layer = nn.Linear(self.fc_configs[-1], self.latent_dim)
        print("Build last layer {} X {}".format(self.fc_configs[-1], self.latent_dim))

    def forward(self, labels, latent_layer_idx=-1, verbose=True):
        """
        G(Z|y) or G(X|y):
        Generate either latent representation( latent_layer_idx < 0) or raw image (latent_layer_idx=0) conditional on labels.
        :param labels:
        :param latent_layer_idx:
            if -1, generate latent representation of the last layer,
            -2 for the 2nd to last layer, 0 for raw images.
        :param verbose: also return the sampled Gaussian noise if verbose = True
        :return: a dictionary of output information.
        """
        result = {}
        batch_size = labels.shape[0]
        eps = torch.rand((batch_size, self.noise_dim)).to(device)
        if verbose:
            result['eps'] = eps
        if self.embedding: # embedded dense vector
            y_input = self.embedding_layer(labels).to(device)
        else: # one-hot (sparse) vector
            y_input = torch.FloatTensor(batch_size, self.n_class).to(device)
            y_input.zero_()
            #labels = labels.view
            y_input.scatter_(1, labels.view(-1,1).to(device), 1).to(device)
        z = torch.cat((eps, y_input), dim=1).to(device)
        ### FC layers
        for layer in self.fc_layers:
            z = layer(z)
        z = self.representation_layer(z).to(device)
        result['output'] = z
        return result

    @staticmethod
    def normalize_images(layer):
        """
        Normalize images into zero-mean and unit-variance.
        """
        mean = layer.mean(dim=(2, 3), keepdim=True)
        std = layer.view((layer.size(0), layer.size(1), -1))             .std(dim=2, keepdim=True).unsqueeze(3)
        return (layer - mean) / std

class DivLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self):
        """
        Class initializer.
        """
        super().__init__()

    def forward2(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        chunk_size = layer.size(0) // 2

        ####### diversity loss ########
        eps1, eps2=torch.split(noises, chunk_size, dim=0)
        chunk1, chunk2=torch.split(layer, chunk_size, dim=0)
        lz=torch.mean(torch.abs(chunk1 - chunk2)) / torch.mean(
            torch.abs(eps1 - eps2))
        eps=1 * 1e-5
        diversity_loss=1 / (lz + eps)
        return diversity_loss

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer=layer.view((layer.size(0), -1))
        chunk_size=layer.size(0) // 2

        ####### diversity loss ########
        eps1, eps2=torch.split(noises, chunk_size, dim=0)
        chunk1, chunk2=torch.split(layer, chunk_size, dim=0)
        lz=torch.mean(torch.abs(chunk1 - chunk2)) / torch.mean(
            torch.abs(eps1 - eps2))
        eps=1 * 1e-5
        diversity_loss=1 / (lz + eps)
        return diversity_loss


class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """
    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))


# In[7]:


generative_model =  Generator(embedding=False, latent_layer_idx=-1)
# generative_model = nn.DataParallel(generative_model).to(device)


# In[8]:


train_dataset,testset, dict_users, dict_users_test = partition_data(n_users = args.num_clients, alpha=0.3,rand_seed = 0, dataset='CIFAR10')


# In[9]:


Loaders_train = LocalDataloaders(train_dataset,dict_users,args.batch_size,ShuffleorNot = True,frac=0.1)
Major_classes = []
Counts = []
Available_labels = []
for idx in range(args.num_clients):
    available_labels = []
    counts = [0]*10
    for batch_idx,(X,y) in enumerate(Loaders_train[idx]):
        batch = len(y)
        y = np.array(y)
        for i in range(batch):
            counts[int(y[i])] += 1
    print(counts)
    Counts.append(counts)
    for i in range(10):
        if counts[i] != 0: available_labels.append(i)
    Available_labels.append(available_labels)


# In[10]:


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
    


# In[11]:


logger = SummaryWriter('./logs')
checkpoint_dir = './checkpoint/'+ args.dataset + '/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
with open(checkpoint_dir+'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)
print('Data and model loaded')
print('Checkpoint dir:', checkpoint_dir)


# In[12]:


for m in global_model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))


# In[13]:


def soft_predict(Z,temp):
    m,n = Z.shape
    Q = torch.zeros(m,n)
    Z_sum = torch.sum(torch.exp(Z/temp),dim=1)
    for i in range(n):
        Q[:,i] = torch.exp(Z[:,i]/temp)/Z_sum
    return Q


# In[14]:


class LocalUpdate(object):
    """
    This class is for train the local model with input global model(copied) and output the updated weight
    args: argument 
    Loader_train,Loader_val,Loaders_test: input for training and inference
    user: the index of local model
    idxs: the index for data of this local model
    logger: log the loss and the process
    """
    def __init__(self, args, Loader_train,available_labels,loader_test,idxs, logger, code_length, num_classes, device):
        self.args = args
        self.logger = logger
        self.trainloader = Loader_train
        self.testloader = loader_test
        self.idxs = idxs
        self.ce = nn.CrossEntropyLoss() 
        self.device = device
        self.code_length = code_length
        self.mse = nn.MSELoss()
        self.model  = net(64,num_classes).to(device)
        self.model = nn.DataParallel(self.model).to(device)
        self.early_stop = 20 
        self.latent_layer_idx = -1
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")
        self.available_labels = available_labels
        self.gen_batch_size = 32
        self.batch_size = 64
        
    def update_weights_Gen(self,generative_model, global_round,regularization=True):
        self.model.to(self.device)
        self.model.train()
        generative_model = generative_model.to(device)
        generative_model.eval()
        epoch_loss = []
        optimizer = optim.Adam(self.model.parameters(),lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                user_output_logp = self.model(X).double()
                predictive_loss = self.ce(user_output_logp,y)    
                
                if regularization and iter < self.early_stop:
                    generative_alpha=self.exp_lr_scheduler(global_round, decay=0.98, init_lr=10)
                    generative_beta=self.exp_lr_scheduler(global_round, decay=0.98, init_lr=10)
                    gen_output=generative_model(y, latent_layer_idx=self.latent_layer_idx)['output']
                    logit_given_gen=self.model.module.classifier(gen_output)
                    target_p=F.softmax(logit_given_gen, dim=1).clone().detach()
                    user_output_logp=F.softmax(user_output_logp, dim=1)
                    user_latent_loss = -generative_beta * self.ensemble_loss(user_output_logp, target_p)
                    
                    sampled_y=np.random.choice(self.available_labels, self.gen_batch_size)
                    sampled_y=torch.tensor(sampled_y).to(device)
                    
                    gen_result=generative_model(sampled_y, latent_layer_idx=-1)
                    gen_output=gen_result['output']
                    user_output_logp =self.model.module.classifier(gen_output).double()
                    user_output_logp=F.softmax(user_output_logp, dim=1)
                    teacher_loss =  self.ce(user_output_logp, sampled_y)
                    loss=predictive_loss + 0.1*teacher_loss + 0.1*user_latent_loss
                
                else:
                    loss=predictive_loss                 
                    
                loss.backward()
#                 nn.utils.clip_grad_norm_(self.model.parameters(), max_norm =1.1)
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)] loss1: {:.6f} loss2: {:.6f} loss3: {:.6f}'.format(
                        global_round, iter, batch_idx * len(X),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), predictive_loss.item(), teacher_loss.item(), user_latent_loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return self.model.state_dict(),sum(epoch_loss) / len(epoch_loss)
   
    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr= max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr
        
    def test_accuracy(self):
        self.model.eval()
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
        


# In[15]:


global_weights = global_model.state_dict()


# In[16]:


# training
args.num_epochs = 50
train_loss, train_accuracy = [], []
val_acc_list, net_list = [], []
cv_loss, cv_acc = [], []
print_every = 2
val_loss_pre, counter = 0, 0
LocalModels = []
loader_test = torch.utils.data.DataLoader(testset, batch_size=64,shuffle=True, num_workers=2)
for idx in range(args.num_clients):
    LocalModels.append(LocalUpdate(args,Loaders_train[idx], Available_labels[idx], Loaders_test[idx], idxs=dict_users[idx], 
                                   logger=logger, code_length = args.code_len, num_classes = 10, device=device))
    


# In[17]:


def get_label_weights():
    label_weights = []
    qualified_labels = []
    for label in range(10):
        weights = []
        for idx in range(args.num_clients):
            weights.append(Counts[idx][label])
        if np.max(weights) > MIN_SAMPLES_PER_LABEL:
            qualified_labels.append(label)
        # uniform
        label_weights.append( np.array(weights) / np.sum(weights) )
    label_weights = np.array(label_weights).reshape((10, -1))
    return label_weights, qualified_labels


# In[18]:


MIN_SAMPLES_PER_LABEL=1
def train_generator(generative_model,global_model,LocalModels,batch_size, epoches=1, latent_layer_idx=-1, verbose=False):
    """
    Learn a generator that find a consensus latent representation z, given a label 'y'.
    :param batch_size:
    :param epoches:
    :param latent_layer_idx: if set to -1 (-2), get latent representation of the last (or 2nd to last) layer.
    :param verbose: print loss information.
    :return: Do not return anything.
    """
    label_weights, qualified_labels = get_label_weights()
    TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS, STUDENT_LOSS2 = 0, 0, 0, 0
    unique_labels = 10
    student_model = global_model
    ensemble_beta = 0
    ensemble_alpha = 1
    ensemble_eta = 1
    n_teacher_iters = 5
    generative_model = generative_model.to(device)
    ce = nn.CrossEntropyLoss() 
    for ep in range(epoches):
        generative_model.train()
        student_model.eval()
        for i in range(n_teacher_iters):
            generative_optimizer = optim.Adam(generative_model.parameters(),lr=args.lr)
            generative_optimizer.zero_grad()
            y=np.random.choice(qualified_labels, batch_size)
            y_input=torch.LongTensor(y).to(device)
            ## feed to generator
            gen_result= generative_model(y_input, latent_layer_idx=latent_layer_idx, verbose=True)
            # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
            gen_output, eps=gen_result['output'].to(device), gen_result['eps'].to(device)
            ##### get losses ####
            diversity_loss = generative_model.diversity_loss(eps, gen_output)  # encourage different outputs

            ######### get teacher loss ############
            teacher_loss=0
            teacher_logit=0
            for user_idx in range(args.num_clients):
                LocalModels[idx].model.eval()
                weight= label_weights[y][:, user_idx].reshape(-1, 1)
                expand_weight=np.tile(weight, (1, unique_labels))
                user_result_given_gen=LocalModels[idx].model.module.classifier(gen_output)
                user_output_logp_=F.log_softmax(user_result_given_gen, dim=1).to(device)
                teacher_loss_=torch.mean(ce(user_output_logp_, y_input) * torch.tensor(weight, dtype=torch.float32).to(device))
                teacher_loss+=teacher_loss_
                teacher_logit+=user_result_given_gen * torch.tensor(expand_weight, dtype=torch.float32).to(device)
            ######### get student loss ############
            
            student_output=student_model.module.classifier(gen_output)
            student_loss=F.kl_div(F.log_softmax(student_output, dim=1), F.softmax(teacher_logit, dim=1))
            if ensemble_beta > 0:
                loss=ensemble_alpha * teacher_loss - ensemble_beta * student_loss + ensemble_eta * diversity_loss
            else:
                loss= ensemble_alpha * teacher_loss + ensemble_eta * diversity_loss

            loss.backward()
            generative_optimizer.step()
            TEACHER_LOSS += ensemble_alpha * teacher_loss     #(torch.mean(TEACHER_LOSS.double())).item()
            STUDENT_LOSS += ensemble_beta * student_loss      #(torch.mean(student_loss.double())).item()
            DIVERSITY_LOSS += ensemble_eta * diversity_loss   #(torch.mean(diversity_loss.double())).item()
            
    return TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS


    TEACHER_LOSS = TEACHER_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
    STUDENT_LOSS = STUDENT_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
    DIVERSITY_LOSS = DIVERSITY_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
    info="Generator: Teacher Loss= {:.4f}, Student Loss= {:.4f}, Diversity Loss = {:.4f}, ".         format(TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)
    if verbose:
        print(info)
    self.generative_lr_scheduler.step()
    


# In[19]:


loader_test = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=False, num_workers=2)

Global_acc = []
Local_acc = []


# In[ ]:


for epoch in tqdm(range(args.num_epochs)):
    test_accuracy = 0

    Knowledges = []
    local_weights, local_losses = [], []
    print(f'\n | Global Training Round : {epoch+1} |\n')
    global_model.train()
    m = max(int(args.sampling_rate * args.num_clients), 1)
    idxs_users = np.random.choice(range(args.num_clients), m, replace=False)
    for idx in idxs_users:
        LocalModels[idx].load_model(global_weights)
        w, loss = LocalModels[idx].update_weights_Gen(generative_model,global_round=epoch, regularization = True)
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
    Local_acc.append(test_accuracy / args.num_clients)
    global_model.load_state_dict(global_weights)
    
    train_generator(generative_model,global_model,LocalModels,32, epoches=1, latent_layer_idx=-1, verbose=True)
    
    accuracy = 0
    cnt = 0
    global_model.eval()
    for cnt, (X,y) in enumerate(loader_test):
        X = X.to(device)
        y = y.double().to(device)
        p = global_model(X)
        y_pred = p.argmax(1).double()
        accuracy += Accuracy(y,y_pred)
        cnt += 1
    print("accuracy of global test:",accuracy/cnt)
    Global_acc.append(accuracy/cnt)


# In[ ]:


print('local test accuracy:')
print(Local_acc)
print('global test accuracy:')
print(Global_acc)

