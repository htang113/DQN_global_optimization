# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 18:32:13 2023

@author: haota
"""

import torch;

def update(data, model, target, optimizer, device, 
           gamma=0.9, steps=10, batch_size = 100):
    
    rewards = torch.tensor(data['reward'], dtype=torch.float).to(device);
    pos_next = torch.tensor(data['next'], dtype=torch.float).to(device);
    aspace = torch.tensor(data['aspace'], dtype=torch.float).to(device);
    pos = torch.tensor(data['pos'], dtype=torch.float).to(device);
    actions = torch.tensor(data['action'], dtype=torch.float).to(device);
    shape = list(aspace.shape);
    
    if(shape[0]>batch_size):
        
        ind = torch.randperm(shape[0])[:batch_size];
        
        rewards = rewards[ind];
        pos_next = pos_next[ind];
        aspace = aspace[ind];
        pos = pos[ind];
        actions = actions[ind];
        shape[0] = batch_size;
    
    pos_next_repeat = torch.repeat_interleave(pos_next, shape[1], dim=0);
    
    aspace   = aspace.reshape([shape[0]*shape[1], shape[2],shape[3]]);
    Qnext = model(pos_next_repeat, aspace).detach();
    Qnext = Qnext.reshape([shape[0], shape[1]]);
    amax = torch.argmax(Qnext, axis=1);
    amax += torch.tensor([shape[1]*i for i in range(shape[0])],dtype=torch.long).to(device);

    Qtarget = target(pos_next, aspace[amax]).detach();

    targets = rewards + gamma*Qtarget;
    
    average_loss = 0;
    
    for _ in range(steps):
        Q = model(pos,actions);
        loss_fn = torch.nn.MSELoss();
        loss = loss_fn(Q, targets);
        
        average_loss += loss;
        
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        
    return average_loss/steps;