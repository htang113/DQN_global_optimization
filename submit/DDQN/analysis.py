#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:46:43 2023

@author: ubuntu
"""

import numpy as np;
import json;
import matplotlib.pyplot as plt;
from actions import action_space;
from env import step;
from generate_local_min import generate_local_minimum;

# =============================================================================
# Ntraj = 185;
# 
# R = [];
# 
# for i in range(Ntraj):
#     with open('data/trajectory_'+str(i)+'.json', 'r') as file:
#         
#         data = json.load(file);
#         rewards = np.mean(data['reward']);
#         R.append(rewards);
#     
# plt.plot(R)
# =============================================================================

init = ''

for _ in range(10):
    N_particles = 20;
    N_actions = 5;
    energy_filter_percentage = 60;
    with open('minima.json','r') as file:
        Eg = json.load(file)[str(N_particles)]['energy'];
    
    pos, energy = generate_local_minimum(N_particles,
                                     energy_filter=Eg*energy_filter_percentage/100);
    
    pos += 0.001*np.random.rand(len(pos),3)
    aspace = action_space(pos,N_actions);
    action = aspace[0];
    action *= 1-2*(action[0,0]<0)
    pos, energy, converge = step(pos,action, max_steps=200);
    
    print(converge)








