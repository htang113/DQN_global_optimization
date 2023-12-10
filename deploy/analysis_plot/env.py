# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 13:28:48 2023

@author: haota
"""

import numpy as np;
from scipy.optimize import minimize;

def H(pos1):

    N = int(len(pos1)//3)
    pos = pos1.reshape([N,3]);
    
    r_vec = pos[:,None,:]-pos[None,:,:];
    dist = np.linalg.norm(r_vec, axis=2)+np.eye(N);
    
    mat1 = 24*(2/dist**14 - 1/dist**8);
    mat2 = 24*(-28/dist**14 + 8/dist**8);
    mat1 = mat1[:,:,None,None]*np.eye(3)[None,None,:,:];
    mat2 = mat2[:,:,None,None]*r_vec[:,:,:,None]*r_vec[:,:,None,:];
    
    Hessian = mat1+mat2;
    for i in range(N):
        Hessian[i,i] -= np.sum(Hessian[i], axis=0); 
        
    Hessian = np.transpose(Hessian, (0,2,1,3));
    Hessian = Hessian.reshape([3*N,3*N]);
    return Hessian;

def E(pos1):
    
    N = int(len(pos1)//3)
    pos = pos1.reshape([N,3]);
    
    r_vec = pos[:,None,:]-pos[None,:,:];
    dist = np.linalg.norm(r_vec, axis=2)+np.eye(N);
    energy = 2*(1/dist**12-1/dist**6);
    
    return np.sum(energy);

def f(pos1):
    
    N = int(len(pos1)//3)
    pos = pos1.reshape([N,3]);
    
    r_vec = pos[:,None,:]-pos[None,:,:];
    dist = np.linalg.norm(r_vec, axis=2)+np.eye(N);
    f = 24*(-2/dist**14+1/dist**8)[:,:,None]*r_vec;
    
    return np.sum(f, axis=1).reshape(3*N);
    
def f_eff(pos1):
    
    f0 = f(pos1);
    H0 = H(pos1);
    el = np.linalg.eigh(H0);
    if(el[0][0]<-0.01):
        e0 = el[1][0];
    else:
        e0 = el[1][6];
        
    return f0 - 2*f0.dot(e0)*e0;
    
    
def step(pos, action, displacement=0.1, accuracy = 1E-4, max_steps=200):
    
    N = len(pos);
    pos = pos.reshape(3*N)
    action = action.reshape(3*N);
    pos0 = pos.copy();
    energy0 = E(pos);

    dE = 1;
    converge = False;
    for _ in range(max_steps):
        pos += action*displacement;
        
        for _ in range(max_steps):
            force = f(pos);
            pos -= 0.001*(force-force.dot(action)*action);
            if(np.linalg.norm(force)<0.1):
                break;
                
        energy = E(pos);
        dE = energy - energy0;
        energy0 = energy;

        if(dE<0):
            converge = True;
            break;
            
    # relax to the next local minimum
    if(converge):
        
        pos += displacement*(pos-pos0)/np.linalg.norm(pos-pos0);
        res = minimize(E, pos, method='Newton-CG', jac=f, hess=H, 
                       tol=accuracy);
        pos = res.x;
        energy = E(pos);
        converge = res.success;
            
    if(not converge):
        
        pos = pos0;
        energy = energy0;
    
    return pos.reshape([N,3]), energy, converge;