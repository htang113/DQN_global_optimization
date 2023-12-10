import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.vibrations import Vibrations
from env import H;

def action_space(pos, M):
    '''
    params:
    pos: a (N, 3) array for N atom positions
    M: pick lowest M eigenmodes from the (3N*3N) Hessian matrix

    returns:
    energy: current potential energy;
    m_eigvecs: (2M, 3*N) array: lowest M eigenvectors * 2(reversed direction)
    '''
    N = len(pos)
    
    Hessian = H(pos.reshape(3*N));
    
    vals, vecs = np.linalg.eigh(Hessian)
    pairs = zip(vals, vecs.T)

    # sort the vectors according to the eigvals
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    lowest_m_pairs = sorted_pairs[6:6+M]
        
    # Separate the eigenvalues and eigenvectors
    m_eigvals = np.array([pair[0] for pair in lowest_m_pairs])
    m_eigvecs = np.array([pair[1].reshape([N,3]) for pair in lowest_m_pairs])

    return np.concatenate((m_eigvecs, -1*m_eigvecs), axis=0)





