# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 21:26:46 2023

@author: haota
"""

import numpy as np;
import ase;
from ase.neb import NEB
from ase.optimize import MDMin,BFGS, FIRE
from ase import Atoms;
from ase.calculators.lj import LennardJones
import json;
from env import step;
from ase import io;

N = 11;
atoms_list = [];

with open('trajectory_DDQN.json','r') as file:
    data = json.load(file);

l1 = [i for i in range(51,56)]
for li in l1:
    pos = data['pos'][li]
    atoms_list.append(Atoms('Fe'+str(len(pos)),
                            cell = [10,10,10],
                            positions=np.array(pos)-np.mean(pos,axis=0)+5));

action = np.array(data['action'][55]);
pos_next, energy, converge = step(np.array(pos),action, max_steps=200);
atoms_list.append(Atoms('Fe'+str(len(pos)),
                        cell = [10,10,10],
                        positions=np.array(pos_next)-np.mean(pos_next,axis=0)+5));

for i in range(len(atoms_list)):
    atoms_list[i].calc = LennardJones(rc = 100);
    
    dyn = BFGS(atoms_list[i]);
    dyn.run(fmax=0.01,steps = 200);

images = [];
for i in range(len(atoms_list)-1):
    image_k = [atoms_list[i].copy()];
    image_k += [atoms_list[i].copy() for u in range(N-2)];
    image_k += [atoms_list[i+1].copy()];

    neb = NEB(image_k)
    neb.interpolate()
    for image in image_k:
        image.calc = LennardJones(rc = 100);
    optimizer = FIRE(neb)

    res = optimizer.run(fmax=0.05,steps = 500);
    images += image_k[:-1];
images += [image_k[-1]];
neb = NEB(images);

for image in images:
    image.calc = LennardJones(rc = 100);
  
optimizer = FIRE(neb)

#res = optimizer.run(fmax=0.05,steps = 1000);

with open('Fig4a.txt','w') as file:
    for i in range(len(images)):
        file.write(str(i/10)+'\t'+str(images[i].get_potential_energy())+'\n');

io.write('XDATCAR_Fig4b',images, format='vasp-xdatcar')

