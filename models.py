import pennylane as qml
from pennylane import numpy as np
import numpy as nnp
import torch.nn as nn
import torch
import math as m
import torch.nn as nn
from tqdm import trange
from tqdm import tqdm
from collections.abc import Iterable
import functools
import os

class Model1(nn.Module):
    def __init__(self,qnode,nq,nl,n_epoch):
        super(Model1, self).__init__()
        
        self.nq = nq
        self.nl = nl
        self.qlayer = qnode
        
        ############### initialization layer ######################3
        
        init_method = functools.partial(torch.nn.init.uniform_, b=2 * m.pi)

        self.alfa =  torch.nn.Parameter(init_method(torch.Tensor(1,4)))

        self.l1 = nn.Linear(4,10)
        self.l2 = nn.Linear(10,self.nq*self.nl)
        self.f = nn.Tanh()

        self.epochs_ = 1
        self.n_epoch = n_epoch

        if not os.path.exists('./Model1_nq_{}_nl_{}_n_{}'.format(nq,nl,self.n_epoch)):
            os.mkdir('./Model1_nq_{}_nl_{}_n_{}'.format(nq,nl,self.n_epoch))

        

    def forward(self,x):
        
        y = self.l1(self.alfa)
        y = self.f(y)
        y = self.l2(y)
        y = self.f(y)
        y = y.reshape(self.nq*self.nl)
        np.savetxt('Model1_nq_{}_nl_{}_n_{}/theta_Net_nq_igual_nl_epoch_{}.txt'.format(self.nq,self.nl,self.n_epoch,self.epochs_),y.detach().numpy())
        self.epochs_+=1
        y = self.qlayer(x,y)
        
        return y
        
   

class Model2(nn.Module):
    def __init__(self,qnode,nq,nl,n_epoch):
        super(Model2, self).__init__()

        self.nq = nq
        self.nl = nl
        self.qlayer = qnode

        ############### initialization layer ######################3

        init_method = functools.partial(torch.nn.init.uniform_, b=2 * m.pi)

        self.alfa =  torch.nn.Parameter(init_method(torch.Tensor(1,4)))

        self.l1 = nn.Linear(4,30)
        self.l2 = nn.Linear(30,self.nq*self.nl)
        self.f = nn.Tanh()

        self.epochs_ = 1
        self.n_epoch = n_epoch

        if not os.path.exists('./Model2_nq_{}_nl_{}_n_{}'.format(nq,nl,self.n_epoch)):
            os.mkdir('./Model2_nq_{}_nl_{}_n_{}'.format(nq,nl,self.n_epoch))


    def forward(self,x):

        y = self.l1(self.alfa)
        y = self.f(y)
        y = self.l2(y)
        y = self.f(y)
        y = y.reshape(self.nq*self.nl)
        np.savetxt('Model2_nq_{}_nl_{}_n_{}/theta_Net_nq_igual_nl_epoch_{}.txt'.format(self.nq,self.nl,self.n_epoch,self.epochs_),y.detach().numpy())
        self.epochs_+=1
        y = self.qlayer(x,y)

        return y


class Model3(nn.Module):
    def __init__(self,qnode,nq,nl,n_epoch):
        super(Model3, self).__init__()

        self.nq = nq
        self.nl = nl
        self.qlayer = qnode

        ############### initialization layer ######################3

        init_method = functools.partial(torch.nn.init.uniform_, b=2 * m.pi)

        self.alfa =  torch.nn.Parameter(init_method(torch.Tensor(1,4)))

        self.l1 = nn.Linear(4,10)
        self.l2 = nn.Linear(10,20)
        self.l3 = nn.Linear(20,self.nq*self.nl)
        self.f = nn.Tanh()

        self.epochs_ = 1
        self.n_epoch = n_epoch

        if not os.path.exists('./Model3_nq_{}_nl_{}_n_{}'.format(nq,nl,self.n_epoch)):
            os.mkdir('./Model3_nq_{}_nl_{}_n_{}'.format(nq,nl,self.n_epoch))


    def forward(self,x):

        y = self.l1(self.alfa)
        y = self.f(y)
        y = self.l2(y)
        y = self.f(y)
        y = self.l3(y)
        y = self.f(y)
        y = y.reshape(self.nq*self.nl)
        np.savetxt('Model3_nq_{}_nl_{}_n_{}/theta_Net_nq_igual_nl_epoch_{}.txt'.format(self.nq,self.nl,self.n_epoch,self.epochs_),y.detach().numpy())
        self.epochs_+=1
        y = self.qlayer(x,y)

        return y




class Net(nn.Module):
    def __init__(self,qnode,nq,nl):
        super(Net, self).__init__()

        self.nq = nq
        self.nl = nl


        weight_shapes = {"w": (nq*nl)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

        
       

    def forward(self,x):

        y = self.qlayer(x)

        return y
