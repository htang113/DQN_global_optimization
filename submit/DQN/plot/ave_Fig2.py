# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 23:09:28 2023

@author: haota
"""

import numpy as np;
import matplotlib.pyplot as plt;
import matplotlib;

R = [];
Steps = [];
rate = [];
trainl = range(6);
N_train = len(trainl);

for u in trainl:
    with open('log'+str(u),'r') as file:
        data = file.readlines();
    
    r = [];
    s = [];
    success = [];
    for i in range(1,1201):
        
        res = data[i][:-1].split();
        r.append(float(res[-7][:-1]));
        s.append(float(res[-1]));
        success.append(res[-4][:-1]=='True');
        
    R.append(r);
    Steps.append(s);
    rate.append(success);
    
R = np.array(R);
interval = 200;
labelsize = 46;
Num = len(r)-interval;
ave = np.zeros([N_train,Num]);
for i in range(Num):
    ave[:,i] = (np.mean(R[:,i:i+interval], axis=1));
    
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
matplotlib.rcParams.update({'font.size': 46})

plt.figure(figsize=(40,12));
plt.subplot(1,3,1)
for i in range(N_train):
    plt.plot(range(len(ave[i])), ave[i],linewidth=1/2,c='grey')

mean = np.mean(ave, axis=0);
var = np.sqrt(np.var(ave,axis=0));
plt.plot(range(len(mean)), mean, c='red', linewidth=3);
#plt.plot(range(len(mean)), mean-var, c='red',linestyle='dashed', linewidth=3);
#plt.plot(range(len(mean)), mean+var, c='red',linestyle='dashed', linewidth=3);
plt.xlabel('epoch',fontdict={'fontsize':labelsize})
plt.ylabel('reward',fontdict={'fontsize':labelsize})
plt.axis([0,1000,1.5,2.7])

plt.subplot(1,3,2)
Steps = np.array(Steps);
ave = np.zeros([N_train,Num]);
for i in range(Num):
    ave[:,i] = (np.mean(Steps[:,i:i+interval], axis=1));

for i in range(N_train):
    plt.plot(range(len(ave[i])), ave[i],linewidth=1/2,c='grey')

mean = np.mean(ave, axis=0);
var = np.sqrt(np.var(ave,axis=0));
plt.plot(range(len(mean)), mean, c='blue', linewidth=3);
#plt.plot(range(len(mean)), mean-var, c='blue',linestyle='dashed', linewidth=3);
#plt.plot(range(len(mean)), mean+var, c='blue',linestyle='dashed', linewidth=3);
plt.xlabel('epoch',fontdict={'fontsize':labelsize})
plt.ylabel('steps',fontdict={'fontsize':labelsize})
plt.axis([0,1000,10,14])

plt.subplot(1,3,3)
rate = np.array(rate);
ave = np.zeros([N_train,Num]);
for i in range(Num):
    ave[:,i] = (np.mean(rate[:,i:i+interval], axis=1));

for i in range(N_train):
    plt.plot(range(len(ave[i])), ave[i],linewidth=1/2,c='grey')

mean = np.mean(ave, axis=0);
var = np.sqrt(np.var(ave,axis=0));
plt.plot(range(len(mean)), mean, c='green', linewidth=3);
#plt.plot(range(len(mean)), mean-var, c='green',linestyle='dashed', linewidth=3);
#plt.plot(range(len(mean)), mean+var, c='green',linestyle='dashed', linewidth=3);
plt.xlabel('epoch',fontdict={'fontsize':labelsize})
plt.ylabel('success rate',fontdict={'fontsize':labelsize})
plt.axis([0,1000,0.6,0.88])
plt.tight_layout()