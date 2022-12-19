import pennylane as qml
from pennylane import numpy as np
import numpy as nnp
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from tqdm import trange
import math as m
import torch.optim as optim


from torch.autograd import Function

import torch.nn as nn
import torch.nn.functional as F



from tqdm import trange
from tqdm import tqdm
from time import sleep

from collections.abc import Iterable
import functools

class Model1(nn.Module):
    def __init__(self,qnode,nq,nl):
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
        
    def forward(self,x):
        
        y = self.l1(self.alfa)
        y = self.f(y)
        y = self.l2(y)
        y = self.f(y)
        y = y.reshape(self.nq*self.nl)
        y = self.qlayer(x,y)
        
        return y
        
   

class Model2(nn.Module):
    def __init__(self,qnode,nq,nl):
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

    def forward(self,x):

        y = self.l1(self.alfa)
        y = self.f(y)
        y = self.l2(y)
        y = self.f(y)
        y = y.reshape(self.nq*self.nl)
        y = self.qlayer(x,y)

        return y


class Model3(nn.Module):
    def __init__(self,qnode,nq,nl):
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

    def forward(self,x):

        y = self.l1(self.alfa)
        y = self.f(y)
        y = self.l2(y)
        y = self.f(y)
        y = self.l3(y)
        y = self.f(y)
        y = y.reshape(self.nq*self.nl)
        y = self.qlayer(x,y)

        return y




class Net(nn.Module):
    def __init__(self,qnode,nq,nl):
        super(Net, self).__init__()

        self.nq = nq
        self.nl = nl


        weight_shapes = {"w": (nq*nl)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

        ############### initialization layer ######################3

    def forward(self,x):

        y = self.qlayer(x)

        return y


def qlayer(ni,nl):

    dev = qml.device('default.qubit',wires=ni)
    @qml.qnode(dev, interface="torch")
    def f(inputs,w):
        for i in range(ni):
            qml.RY( inputs[i], wires=i )

        for i in range(nl):
            for j in range(ni):
                qml.RY( w[i*ni+j],wires=j )
            for j in range(ni-1):
                qml.CNOT(wires=[j,j+1])

        return [ qml.probs(i) for i in range(ni) ]
    return f




def cost(out,nq):
    s = 0
    for i in range(0,len(out),2):
        s+=out[i]
    return 1-s/nq



def main(nqMin,nqMax,n):
    nMax = 20e4
    
    dx = np.arange(nqMin,nqMax)
    
    ########################################## Meta #######################
    dz1 = []
    for jj in range(1,n+1):
        dy = []
        for ni in range(nqMin,nqMax):

            nl = ni
            f = qlayer(ni,nl)
            model = MetaLayer(f,ni,nl)
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            x = torch.ones(ni)*(m.pi/4)

            ex = 1
            dn = 1
            while ex>=0.001:
                optimizer.zero_grad()

                out = model(x)

                l = cost(out,ni)
                l.backward()

                optimizer.step()

                ex = l.item()
                if dn%100==0:
                    print('Meta','n:',jj ,'nq:' , ni,':', dn ,'out:', l.item())

                if dn>=nMax:
                    ex = 0.0
                    dn = 0
                    break
                else:
                    dn+=1


            #dx.append(ni)
            dy.append(dn)
            del model
        dz1.append(dy)

    dz1 = np.array(dz1)
    np.savetxt('numMeta2_nq_igual_nl.txt',dz1)

    ########################################## net #######################
    dz = []
    for jj in range(1,n+1):
        dy = []
        for ni in range(nqMin,nqMax):

            nl = ni
            f = qlayer(ni,nl)
            model = Net(f,ni,nl)
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            x = torch.ones(ni)*(m.pi/4)

            ex = 1
            dn = 1
            while ex>=0.001:
                optimizer.zero_grad()

                out = model(x)

                l = cost(out,ni)
                l.backward()

                optimizer.step()
                
                ex = l.item()
                if dn%100==0:
                    print('Net','n:',jj ,'nq:' , ni,':', dn  ,'out:', l.item())

                if dn>=nMax:
                    ex = 0.0
                    dn = 0
                    break
                else:
                    dn+=1



            #dx.append(ni)
            dy.append(dn)
            del model
        dz.append(dy)

    dz = np.array(dz)


    np.savetxt('numNet.txt',dz)


def main_3(nqMin,nqMax,nl,n):
    nMax = 30e4
    Cost_ = 0.3
    
    ########################################## net #######################
    dz = []
    for jj in range(1,n+1):
        dy = []
        for nq in range(nqMin,nqMax,2):

            f = qlayer(nq,nl)
            model = Net(f,nq,nl)
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            x = torch.ones(nq)*(m.pi/4)

            ex = 1
            dn = 1
            while ex>=Cost_:
                optimizer.zero_grad()

                out = model(x)

                l = cost(out,nq)
                l.backward()

                optimizer.step()

                ex = l.item()
                if dn%100==0:
                    print('Net','n:',jj ,'nq:' , nq,':', dn  ,'out:', l.item())

                if dn>=nMax:
                    ex = 0.0
                    dn = 0
                    break
                else:
                    dn+=1




            dy.append(dn)
            del model
        dz.append(dy)

    dz = np.array(dz)

    
    ########################################## Model 1 #######################
    dz1 = []
    for jj in range(1,n+1):
        dy = []
        for nq in range(nqMin,nqMax,2):

            f = qlayer(nq,nl)
            model = Model1(f,nq,nl)
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            x = torch.ones(nq)*(m.pi/4)

            ex = 1
            dn = 1
            while ex>=Cost_:
                optimizer.zero_grad()

                out = model(x)

                l = cost(out,nq)
                l.backward()

                optimizer.step()

                ex = l.item()
                if dn%100==0:
                    print('Model 1','n:',jj ,'nq:' , nq,':', dn ,'out:', l.item())

                if dn>=nMax:
                    ex = 0.0
                    dn = 0
                    break
                else:
                    dn+=1


            #dx.append(ni)
            dy.append(dn)
            del model
        dz1.append(dy)

    dz1 = np.array(dz1)
    np.savetxt('numMeta1_nl_{}.txt'.format(nl),dz1)

    ########################################## Model 2 #######################
    dz2 = []
    for jj in range(1,n+1):
        dy = []
        for nq in range(nqMin,nqMax,2):

            f = qlayer(nq,nl)
            model = Model2(f,nq,nl)
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            x = torch.ones(nq)*(m.pi/4)

            ex = 1
            dn = 1
            while ex>=Cost_:
                optimizer.zero_grad()

                out = model(x)

                l = cost(out,nq)
                l.backward()

                optimizer.step()

                ex = l.item()
                if dn%100==0:
                    print('Model 2','n:',jj ,'nq:' , nq,':', dn ,'out:', l.item())

                if dn>=nMax:
                    ex = 0.0
                    dn = 0
                    break
                else:
                    dn+=1


            #dx.append(ni)
            dy.append(dn)
            del model
        dz2.append(dy)

    dz2 = np.array(dz2)
    np.savetxt('numMeta2_nl_{}.txt'.format(nl),dz2)

    ########################################## Model 3 #######################
    dz3 = []
    for jj in range(1,n+1):
        dy = []
        for nq in range(nqMin,nqMax,2):

            f = qlayer(nq,nl)
            model = Model3(f,nq,nl)
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            x = torch.ones(nq)*(m.pi/4)

            ex = 1
            dn = 1
            while ex>=Cost_:
                optimizer.zero_grad()

                out = model(x)

                l = cost(out,nq)
                l.backward()

                optimizer.step()

                ex = l.item()
                if dn%100==0:
                    print('Model 3','n:',jj ,'nq:' , nq,':', dn ,'out:', l.item())

                if dn>=nMax:
                    ex = 0.0
                    dn = 0
                    break
                else:
                    dn+=1


            #dx.append(ni)
            dy.append(dn)
            del model
        dz3.append(dy)

    dz3 = np.array(dz3)
    np.savetxt('numMeta3_nl_{}.txt'.format(nl),dz3)
    

    



#main(nqMin,nqMax,n)

#main(2,11,10)
#main_3(nqMin,nqMax,nl,n)
main_3(2,12,100,10)












