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
    Nepoch = 1500;
    time_horizon = 20;
    update_target = 10;
    T0 = 3;
    T_decay = 0.998;
    batch_size = 300;
    steps = 20;
    terminal_reward_percentage = 3;
    energy_filter_percentage = 60;
     
    model_filename = 'model'+str(task)+'.pt';
    
    model = Q_net(device).to(device);
    target = Q_net(device).to(device);
    target.load_state_dict(model.state_dict());
    optimizer = torch.optim.SGD(model.parameters(), lr=1E-2);
    
    with open('minima.json','r') as file:
        Eg = json.load(file)[str(N_particles)]['energy'];
    with open('log'+str(task),'w') as file:
        file.write('training log\n')
    
    np.random.seed(task**4*100);

    for epoch in range(Nepoch):    
        
        pos, energy = generate_local_minimum(N_particles,
                                         energy_filter=Eg*energy_filter_percentage/100);
        
        for tstep in range(time_horizon):
            aspace = action_space(pos,N_actions);
            data['pos'].append(pos.tolist());
            data['energy'].append(energy);
            
            aspace = aspace.tolist();
            
            converge = False;
            
            Q = model(torch.tensor([pos.tolist()]*len(aspace)).to(device),
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
            

        if(epoch%update_target == 0):
            
            target.load_state_dict(model.state_dict());
        
        if(epoch%100 == 0):
            torch.save(model.state_dict(), str(epoch)+model_filename);
        
        if(epoch>=100):
            loss = update(data, model, target, optimizer, device,
                          gamma=gamma, steps=steps,batch_size=batch_size);
        else:
            loss = 0;
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
    
    npal = 4;
    inputs = [i for i in range(npal)];
    
    with mp.Pool(processes=npal) as pool:
        
        results = pool.map(run, inputs);

