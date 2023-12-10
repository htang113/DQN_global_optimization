# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:03:00 2023

@author: haota
"""

import json;
import numpy as np;
import matplotlib.pyplot as plt;
import matplotlib;

N_particles = 20;

namel = ['DQN','DDQN','random'];
matplotlib.rcParams.update({'font.size': 46})
plt.figure(figsize=(18,12));
for name in namel:
    with open('log_'+name,'r') as file:
        data = file.readlines();
    
    r = [];
    s = [];
    success = [];
    for i in range(1,201):
        
        res = data[i][:-1].split();
        r.append(float(res[-7][:-1]));
        s.append(float(res[-1]));
        success.append(res[-4][:-1]=='True');
    
    with open('trajectory_'+name+'.json','r') as file:
        data = json.load(file);
    
    with open('minima.json','r') as file:
        Eg = json.load(file)[str(N_particles)]['energy'];
    
    Earray = [];
    
    i_ind = 0;
    for i in range(200):
        
        energy = data['energy'][i_ind:i_ind+int(s[i])];
        energy += [Eg]*(20-len(energy));
        Earray.append(energy);
        i_ind += int(s[i]);
        
    
    del Earray[-1];
    Earray1 = np.array(Earray);
    Earray1 = np.sort(Earray1, axis=0);
    
    mean = np.mean(Earray1, axis=0);
    std = np.sqrt(np.var(Earray1, axis=0));
    
    if(name=='DQN'):
        c1 = 'blue';
    elif(name=='DDQN'):
        c1='red';
    else:
        c1='black'
    
    plt.plot(range(20), np.log(mean-Eg),
             c=c1, label=name, linewidth=4);
#    plt.plot(range(20), np.log(mean+std-Eg),c=c1,linestyle='dashed', linewidth=4);
    lower =mean-std-Eg;
    lower = lower*(lower>0) + 0.025*(lower<0);
#    plt.plot(range(20), np.log(lower),c=c1,linestyle='dashed', linewidth=4);
    plt.axis([0,19,-0.8,1.5])

    plt.xlabel('time steps');
    plt.ylabel('log(<E-Eg>)')
plt.legend()