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

def run(task):
    
    device = 'cuda';
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
    Nepoch = 1500;
    time_horizon = 20;
    save_model = 100;
    T0 = 3;
    T_decay = 0.998;
    batch_size = 300;
    steps = 10;
    terminal_reward_percentage = 3;
    energy_filter_percentage = 60;
    
    model_filename = 'model'+str(task)+'.pt';
    
    modelA = Q_net(device).to(device);
    modelB = Q_net(device).to(device);
    modelB.load_state_dict(modelA.state_dict());
    
    optimizerA = torch.optim.SGD(modelA.parameters(), lr=1E-2);
    optimizerB = torch.optim.SGD(modelB.parameters(), lr=1E-2);
    
    N_particles = 20;
    with open('minima.json','r') as file:
        Eg = json.load(file)[str(N_particles)]['energy'];
    
    with open('log'+str(task),'w') as file:
        file.write('training log\n')
    
    for epoch in range(Nepoch):
        
        np.random.seed(task*10000 + epoch);
        pos, energy = generate_local_minimum(N_particles,
                                         energy_filter=Eg*energy_filter_percentage/100);
        
        for tstep in range(time_horizon):
            
            aspace = action_space(pos,N_actions);
            data['pos'].append(pos.tolist());
            data['energy'].append(energy);
            
            aspace = aspace.tolist();
            
            converge = False;
            if(epoch%2 == 0):
                Q = modelA(torch.tensor([pos.tolist()]*len(aspace)).to(device),
                        torch.tensor(aspace).to(device));
            
            else:
                Q = modelB(torch.tensor([pos.tolist()]*len(aspace)).to(device),
                        torch.tensor(aspace).to(device));
            
            T = T0*T_decay**epoch;
            probs = nn.Softmax(dim=0)(Q/T);
            sampler = Categorical(probs);
            action_id = sampler.sample();
            action = np.array(aspace[action_id]);
            pos, energy, converge = step(pos,action, max_steps=200);

            data['action'].append(action.tolist());
            
            aspace = action_space(pos,N_actions);
            data['aspace'].append(aspace.tolist());
            data['reward'].append(data['energy'][-1]-energy);
            data['next'].append(pos.tolist());
            
            if(abs(energy-Eg)<1E-2):
                
                data['reward'][-1] += - energy*terminal_reward_percentage/100;
                
                break;
            

        if(epoch%save_model == 0):
            
            torch.save(modelA.state_dict(), str(epoch)+model_filename);
        
        for _ in range(10):
            if(epoch%2 == 0):
                loss = update(data, modelA, modelB, optimizerA, device,
                            gamma=gamma, steps=steps,batch_size=batch_size);
            else:
                loss = update(data, modelB, modelA, optimizerB, device,
                            gamma=gamma, steps=steps,batch_size=batch_size);                

        data['loss'].append(float(loss));
    
        
        R = sum([data['reward'][-u1-1]*gamma**(tstep-u1) for u1 in range(tstep+1)]);
        outstr = 'Epoch '+str(epoch)+':  loss='+str(float(loss))+', rewards = '+str(R);
        outstr += ', success = ' +str(abs(energy-Eg)<1E-2);
        outstr += ', tstep = ' + str(tstep+1);
        print(outstr);
        with open('log'+str(task),'a') as file:
            file.write(outstr+'\n');
            
    with open('trajectory'+str(task)+'.json','w') as file:
        json.dump(data, file);        
    
    return True;
    
if __name__ == '__main__':
    
    run(0);
    run(3);

