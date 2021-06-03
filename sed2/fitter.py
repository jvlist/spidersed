import pymc3 as pymc
import numpy as np
import sed2.fg_models
import pickle
import sys
from sed2.lib import *
from mpi4py import MPI
import json
from sed2.base import MapCollection


def sed_ravel(specs, ells, pols, fl=None, map_coll=None):
    '''
    Reorganize the specs dictionary into lists that MPI can scatter to parallel processes for pymc.
    '''
    polmap = dict(TT=0, EE=1, BB=2, TE=3, EB=4, TB=5)
    
    blist = []
    slist = []
    
    ac = map_coll.all_crosses()
       #fs = np.array([map_coll.cross_freq(m1,m2) for m1, m2 in ac])
       #fs.sort()


    if fl != 'None':
        with open(fl, 'rb') as pfile:
            flf = pickle.load(pfile)
    
    for i, l in enumerate(ells):
        for j, p in enumerate(pols):
            elltest = np.array(next(iter(specs.values()))[0])  # Get some random spec and pull out the ell list. Should be the same for all fs
            try:
                ellind = np.where(elltest == l)[0][0]
            except IndexError:
                raise LookupError('One of the ells_to_do ({}) was not in the list of ells from estimate_spectrum.'.format(l))
            
            polind = polmap[p]
            vals = [specs[(m1,m2)][1][polind][ellind] for (m1,m2) in ac]
            errs = [specs[(m1,m2)][2][polind][ellind] for (m1,m2) in ac]

            if fl != 'None':

                flv = [flf[(m1,m2)][1][polind][ellind] for (m1,m2) in ac]
                fle = [flf[(m1,m2)][2][polind][ellind] for (m1,m2) in ac]

                vals = list(np.divide(vals, flv))
                errs = list(np.divide(errs, fle))
            
            blist.append( (l,p) )
            slist.append( (vals, errs) )

    return blist, slist

def file_write(ell, pol, trace, write_dir):
    wname = write_dir + '/' + '{}_{}_trace.p'.format(ell, pol)
    with open(wname, 'wb') as pfile:
        pickle.dump(trace, pfile)
    

if __name__ == '__main__':
    _, do_sim, model_name, spec_file, map_coll, fl_file, ells, pols, post_dir = sys.argv
    do_sim = do_sim == 'True'
    ells = json.loads(ells)
    pols = json.loads(pols)
    map_coll = MapCollection(json.loads(map_coll))


    # Initialize MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()


    # Grab data in the root process
    if rank == 0:
        #with open(pairs_file, 'rb') as pfile: 
        #   freq_pairs = pickle.load(pfile)

        with open(spec_file, 'rb') as pfile:
            specs = pickle.load(pfile)

        if fl_file != 'None':
            with open(fl_file, 'rb') as pfile: 
                fls = pickle.load(pfile)

        ac = map_coll.all_crosses()
        #fs = np.array([map_coll.cross_freq(m1,m2) for m1, m2 in ac])
        #fs.sort()

        blist, slist = sed_ravel(specs, ells, pols, fl=fl_file, map_coll=map_coll)
        if size != len(blist):
            raise ValueError('Number of MPI processes should be len(ells)*len(pols)')

    else:
        #freq_pairs, fs = None, None
        map_coll, ac = None, None
        blist, slist = None, None
    

    # Process Communication 
    bs = comm.scatter(blist, root=0)  # One ell,pol bin per process
    ss = comm.scatter(slist, root=0)
    
    #freq_pairs = comm.bcast(freq_pairs, root=0)  # Everybody needs freq_pairs and fs
    #fs = comm.bcast(fs, root=0)
    map_coll = comm.bcast(map_coll, root=0)  # Everybody needs freq_pairs and fs
    ac = comm.bcast(ac, root=0)
    

    with add_prefix(f'Process #{rank}'):
        dat, err = ss
        ell, pol = bs
        print('Initializing model.')
        model = sed2.fg_models.init('dustxcmb', dat, err, ac, map_coll, do_sim)

        with model:  # pymc3 wants the model object in the context stack when running things
            # Gradient based step methods are currently bugged, so use Metropolis-Hastings for now. 
            # Also, don't try to split into multiple processes, because it probably won't help given MPI is around
            print('Sampling with model.')
            if rank != 0:
                trace = pymc.sample(5000, tune=1000, step=pymc.Metropolis(), cores=1, progressbar=False, return_inferencedata=False)
            else:
                trace = pymc.sample(5000, tune=1000, step=pymc.Metropolis(), cores=1, progressbar=True, return_inferencedata=False)
            
            file_write(ell, pol, trace, post_dir)
