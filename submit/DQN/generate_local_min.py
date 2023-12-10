from ase.optimize import BFGS
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.constraints import FixCom
import numpy as np;

def generate_local_minimum(N, energy_filter=-15):
    '''
    params:
    pos: a (N, 3) array for N atom positions
    '''

    new_energy = 1;
    while(new_energy>energy_filter):
        
        pos = N**(1/3)*np.random.rand(N, 3);
        atoms = Atoms('Ar'+str(len(pos)), positions=pos)
        atoms.calc = LennardJones(rc = 100)
        
        atoms.set_constraint(FixCom())
    
        # process!
        relax = BFGS(atoms, logfile=None)
        converge = relax.run(fmax=0.001, steps = 200);
    
        # After relaxation, you can get the new positions
        
        if(converge):
            criterion = np.max(atoms.get_all_distances())>5;
            if(criterion):
                new_energy = 1;
            else:
                new_positions = atoms.get_positions();
                new_energy = atoms.get_potential_energy();
        else:
            new_energy = 1;
        #print("Relaxed atomic positions:", new_positions)
#        print("Relaxed potential energy:", new_energy)
    #    view(atoms)

    return new_positions, new_energy
