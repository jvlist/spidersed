import pymc3
import numpy as np
import fg_models
import pickle
import sys
from sed_lib import *
from mpi4py import MPI
import json

_, do_sim, model_name, spec_file, pairs_file, fl_file, ells, pols, post_dir = sys.argv
do_sim = do_sim == 'True'
ells = json.loads(ells)
pols = json.loads(pols)

# Initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()



def sed_ravel(specs, ells, pols):
    '''
    Reorganize the specs dictionary into lists that MPI can scatter to parallel processes for pymc.
    '''
    polmap = dict(TT=0, EE=1, BB=2, TE=3, EB=4, TB=5)
    
    blist = []
    slist = []
    
    for i, l in enumerate(ells):
        for j, p in enumerate(pols):
            elltest = np.array(next(iter(specs.values()))[0])  # Get some random spec and pull out the ell list. Should be the same for all fs
            try:
                ellind = np.where(elltest == l)[0][0]
            except IndexError:
                raise LookupError('One of the ells_to_do ({}) was not in the list of ells from estimate_spectrum.'.format(l))
            
            polind = polmap[p]
            vals = [specs[f][1][polind][ellind] for f in fs]
            errs = [specs[f][2][polind][ellind] for f in fs]

            blist.append( (l,p) )
            slist.append( (vals, errs) )

    return blist, slist

def file_write(ell, pol, trace, write_dir):
    wname = write_dir + '/' + '{}_{}_trace.p'.format(ell, pol)
    with open(wname, 'wb') as pfile:
        pickledump(trace, wname)
    

# Grab data in the root process
if rank == 0:
    with open(pairs_file, 'rb') as pfile: 
        freq_pairs = pickle.load(pfile)

    with open(spec_file, 'rb') as pfile:
        specs = pickle.load(pfile)

    if fl_file != 'None':
        with open(fl_file, 'rb') as pfile: 
            fls = pickle.load(pfile)

    fs = np.array(list(specs.keys()), dtype=float)
    fs.sort()

    blist, slist = sed_ravel(specs, ells, pols)
    assert size == len(blist), 'Number of MPI processes should be len(ells)*len(pols)'
else:
    freq_pairs, fs = None, None
    blist, slist = None, None
    

# Process Communication 
bs = comm.scatter(blist, root=0)  # One ell,pol bin per process
ss = comm.scatter(slist, root=0)

freq_pairs = comm.bcast(freq_pairs, root=0)  # Everybody needs freq_pairs and fs
fs = comm.bcast(fs, root=0)


with add_prefix(f'Process #{rank}'):
    dat, err = ss
    ell, pol = bs
    print('Initializing model.')
    model = fg_models.init('dustxcmb', dat, err, fs, freq_pairs, do_sim)

    with model:  # pymc3 wants the model object in the context stack when running things
        # Gradient based step methods are currently bugged, so use Metropolis-Hastings for now. 
        # Also, don't try to split into multiple processes, because it probably won't help given MPI is around
        trace = pymc.sample(5000, tune=1000, step=pymc.Metropolis(), cores=1, progresbar=False, return_inferencedata=False)
        
        file_write(ell, pol, trace, post_dir)
