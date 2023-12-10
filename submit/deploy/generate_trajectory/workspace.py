# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:17:22 2023

@author: haota
"""

import numpy as np;
import json;
from actions import action_space;
from env import step;
from generate_local_min import generate_local_minimum;
from model import Q_net;
from torch import nn;
from torch.distributions.categorical import Categorical;
import torch;
from DQN import update;
import torch.multiprocessing as mp;

task = 'DDQN';

device = 'cpu';
data = {'pos' : [],
        'energy' : [],
        'action' : [],
        'target' : [],
        'aspace' : [],
        'reward' : [],
        'next': [],
        'loss': []};

gamma = 0.8;
N_actions = 3;
N_particles = 20;
Nepoch = 200;
time_horizon = 20;

T0 = 0.2;
T_decay = 1;

terminal_reward_percentage = 3;
energy_filter_percentage = 60;
 
model = Q_net(device).to(device);
model.load_state_dict(torch.load('model_'+task+'.pt'));

with open('minima.json','r') as file:
    Eg = json.load(file)[str(N_particles)]['energy'];
with open('log'+str(task),'w') as file:
    file.write('training log\n')

for epoch in range(Nepoch):
    
#    np.random.seed(task*10000 + epoch);
    pos, energy = generate_local_minimum(N_particles,
                                     energy_filter=Eg*energy_filter_percentage/100);
    
    converge = True;
    
    for tstep in range(time_horizon):
        aspace = action_space(pos,N_actions);
          
        data['pos'].append(pos.tolist());
        data['energy'].append(energy);
        
        aspace = aspace.tolist();
        
        Q = model(torch.tensor([pos.tolist()]*len(aspace)).to(device),
                  torch.tensor(aspace).to(device));
        T = T0*T_decay**epoch;
        if(not converge):
            T = 2;
            
        probs = nn.Softmax(dim=0)(Q/T);
        sampler = Categorical(probs);
        action_id = sampler.sample();
        action = np.array(aspace[action_id]);
        
        converge = False;
        pos, energy, converge = step(pos,action, max_steps=200);

        data['action'].append(action.tolist());
        
        aspace = action_space(pos,N_actions);
        data['aspace'].append(aspace.tolist());
        data['reward'].append(data['energy'][-1]-energy);
        data['next'].append(pos.tolist());
        
        if(abs(energy-Eg)<1E-2):
            
            data['reward'][-1] += - energy*terminal_reward_percentage/100;
            
            break;
        

 
    loss = 0;
    data['loss'].append(float(loss));

    
    R = sum([data['reward'][-u1-1]*gamma**(tstep-u1) for u1 in range(tstep+1)]);
    outstr = 'Epoch '+str(epoch)+':  loss='+str(float(loss))+', rewards = '+str(R);
    outstr += ', success = ' +str(abs(energy-Eg)<1E-2);
    outstr += ', tstep = ' + str(tstep+1);
    print(outstr);
    with open('log'+str(task),'a') as file:
        file.write(outstr+'\n');
        
with open('trajectory_'+str(task)+'.json','w') as file:
    json.dump(data, file);        
    


