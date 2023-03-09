import numpy as nnp
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from tqdm import trange
import math as m
import torch.optim as optim
import torch.nn as nn
from tqdm import trange
from tqdm import tqdm
from collections.abc import Iterable
import functools
import pennylane as qml
from pennylane import numpy as np

import models as md


######################## quantum model ################################

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


######################### loss function ###############################

def cost(out,nq):
    s = 0
    for i in range(0,len(out),2):
        s+=out[i]
    return 1-s/nq




################################ function util ###################################3


def main(nqMin,nqMax,n):
    nMax = 50e4
    cost_ = 0.001
    
    ########################################## Net #######################
    num_interaction_data = []

    for jj in range(1,n+1):
        num_interaction = []
        tp = trange(nqMin,nqMax,2)
        for ni in tp:

            nl = ni
            f = qlayer(ni,nl)
            model = md.Net(f,ni,nl)
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            x = torch.ones(ni)*(m.pi/4)

            ex = 1
            dn = 1
            loss_hist = []
            while ex>=cost_:
                optimizer.zero_grad()

                out = model(x)

                l = cost(out,ni)
                l.backward()

                optimizer.step()

                ex = l.item()
                tp.set_description(f" Net, nq:{ni} n:{jj}, epoch: {dn}, out: {l.item()}   ")
                  
                if dn>=nMax:
                    break
                else:
                    dn+=1
                loss_hist.append(ex)
            loss_hist = np.array(loss_hist)
            np.savetxt('./loss_data/lossNet_nq_{}_n_{}.txt'.format(ni,jj),loss_hist)



            
            num_interaction.append(dn)
            del model
        num_interaction_data.append(num_interaction)

    num_interaction_data = np.array(num_interaction_data)
    np.savetxt('numNet_nq_igual_nl.txt',num_interaction_data)
    del num_interaction_data
    
    ########################################## Model 1 #######################
    num_interaction_data = []

    for jj in range(1,n+1):
        num_interaction = []
        tp = trange(nqMin,nqMax,2)
        for ni in tp:

            nl = ni
            f = qlayer(ni,nl)
            model = md.Model1(f,ni,nl,jj)
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            x = torch.ones(ni)*(m.pi/4)

            ex = 1
            dn = 1
            loss_hist = []
            while ex>=cost_:
                optimizer.zero_grad()

                out = model(x)

                l = cost(out,ni)
                l.backward()

                optimizer.step()

                ex = l.item()
                tp.set_description(f" Model 1, nq:{ni} n:{jj}, epoch: {dn}, out: {l.item()}   ")
                  
                if dn>=nMax:
                    break
                else:
                    dn+=1
                loss_hist.append(ex)
            loss_hist = np.array(loss_hist)
            np.savetxt('./loss_data/lossModel1_nq_{}_n_{}.txt'.format(ni,jj),loss_hist)



            
            num_interaction.append(dn)
            del model
        num_interaction_data.append(num_interaction)

    num_interaction_data = np.array(num_interaction_data)
    np.savetxt('numModel1_nq_igual_nl.txt',num_interaction_data)
    del num_interaction_data

    ########################################## Model 2 #######################
    num_interaction_data = []

    for jj in range(1,n+1):
        num_interaction = []
        tp = trange(nqMin,nqMax,2)
        for ni in tp:

            nl = ni
            f = qlayer(ni,nl)
            model = md.Model2(f,ni,nl,jj)
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            x = torch.ones(ni)*(m.pi/4)

            ex = 1
            dn = 1
            loss_hist = []
            while ex>=cost_:
                optimizer.zero_grad()

                out = model(x)

                l = cost(out,ni)
                l.backward()

                optimizer.step()

                ex = l.item()
                tp.set_description(f" Model 2, nq:{ni} n:{jj}, epoch: {dn}, out: {l.item()}   ")
                  
                if dn>=nMax:
                    break
                else:
                    dn+=1
                loss_hist.append(ex)
            loss_hist = np.array(loss_hist)
            np.savetxt('./loss_data/lossModel2_nq_{}_n_{}.txt'.format(ni,jj),loss_hist)



            
            num_interaction.append(dn)
            del model
        num_interaction_data.append(num_interaction)

    num_interaction_data = np.array(num_interaction_data)
    np.savetxt('numModel2_nq_igual_nl.txt',num_interaction_data)
    del num_interaction_data

    ########################################## Model 3 #######################
    num_interaction_data = []

    for jj in range(1,n+1):
        num_interaction = []
        tp = trange(nqMin,nqMax,2)
        for ni in tp:

            nl = ni
            f = qlayer(ni,nl)
            model = md.Model3(f,ni,nl,jj)
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            x = torch.ones(ni)*(m.pi/4)

            ex = 1
            dn = 1
            loss_hist = []
            while ex>=cost_:
                optimizer.zero_grad()

                out = model(x)

                l = cost(out,ni)
                l.backward()

                optimizer.step()

                ex = l.item()
                tp.set_description(f" Model 3, nq:{ni} n:{jj}, epoch: {dn}, out: {l.item()}   ")
                  
                if dn>=nMax:
                    break
                else:
                    dn+=1
                loss_hist.append(ex)
            loss_hist = np.array(loss_hist)
            np.savetxt('./loss_data/lossModel3_nq_{}_n_{}.txt'.format(ni,jj),loss_hist)




            
            num_interaction.append(dn)
            del model
        num_interaction_data.append(num_interaction)

    num_interaction_data = np.array(num_interaction_data)
    np.savetxt('numModel3_nq_igual_nl.txt',num_interaction_data)





main(2,12,10) 
