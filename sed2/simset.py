import numpy as np
import pickle
import sys
from sed2.lib import *
from mpi4py import MPI
import json
from sed2.base import MapCollection, SED
import itertools
from os.path import exists
from os import remove
import time
import filelock


def seedstack(specs, dotot, ellb, allcls, jobreqs):
    
    totspec = {}
    
    for i, c in enumerate(allcls):
        ms = jobreqs[i][0]

        if ms in specs.keys():
            specs[ms][1] = np.dstack((specs[ms][1], c))
        else:
            specs[ms] = [ellb, c]
        
    if dotot:
        for k in specs:
            #print(np.shape(specs[k][1]))
            #specs[k][1] = specs[k][1][0]  # Strip off empty third dimension that gets added by dstack
            totspec[k] = [specs[k][0], np.median(specs[k][1], axis=2), nanmad(specs[k][1], axis=2)]
            #print(np.shape(totspec[k][1]))

    return specs, totspec


if __name__ == '__main__':
    _, map_coll, map_dir, fl_file, seeds, noise_dict, auto_suffixes, spec_err_file, pairlist, ac, sig = sys.argv
    with open('./tmpmask', 'rb') as pfile:
        mask = pickle.load(pfile)
    
    map_dir = map_dir.format(sig)

    pairlist = json.loads(pairlist)
    ac = json.loads(ac)
    ac = list(map(tuple, ac))

    seeds = json.loads(seeds)
    map_coll = MapCollection(json.loads(map_coll))
    noise_map = json.loads(noise_dict)
    auto_suffixes = json.loads(auto_suffixes)


    # Initialize MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Grab data in the root process
    if rank == 0:
        
        if fl_file != 'None':
            with open(fl_file, 'rb') as pfile: 
                fls = pickle.load(pfile)

        #ac = map_coll.all_crosses()

        #pairs = itertools.permutations(seeds, 2)

        jobreqs = [(cross, pair) for cross in ac for pair in pairlist]

    else:
        
        jobreqs = None

    #print(jobreqs)
    jr = comm.scatter(jobreqs, root=0)

    m1, m2 = jr[0]
    s1, s2 = jr[1]

    f1, f2 = 0, 0  # These are holdovers from old code and don't actually do anything
    print(f'{m1} seed {s1}  x  {m2} seed {s2} (RANK {rank}) STARTED')
    
    
    mfile1 = map_dir+m1
    mfile2 = map_dir+m2
    
    nfile1 = map_dir + noise_map[m1].format(s1)
    nfile2 = map_dir + noise_map[m2].format(s2)

    if mfile1 == mfile2:
        ext1 = auto_suffixes[0]
        ext2 = auto_suffixes[1]
    else:
        ext1, ext2 = '', ''
        
    ellb, clsb, clse, freq = cross_maps(mask, True, mfile1, mfile2, 
                                        f1, f2, ext1, ext2, 
                                        noisefile1=nfile1, noisefile2=nfile2,
                                    )


    allcls = comm.gather(clsb, root=0)

    if rank == 0:
        '''
        count = -1  # Start with impossible to satisfy condition
        while count != playnice:  # Wait your turn!
            with open('./errorcounter', 'r') as f:
                count = int(f.read())
            time.sleep(5)
        '''
        lock = filelock.FileLock('./lock', timeout=60)

        with lock:

            if exists(spec_err_file+'.full'):
                with open(spec_err_file+'.full', 'rb') as pfile:
                    specs = pickle.load(pfile)
            else:
                specs = {}

            with open('./errorcounter', 'r') as f:
                count = int(f.read())

            dotot = not count

            if dotot:
                print('I am the last process. Running statsitics.')

            #print(np.shape(allcls))

            specs, totspec = seedstack(specs, dotot, ellb, allcls, jobreqs)
            
            with open(spec_err_file+'.full', 'wb') as pfile:
                pickle.dump(specs, pfile)
            
            if dotot:
                with open(spec_err_file, 'wb') as pfile:
                    pickle.dump(totspec, pfile)
            else:
                count -= 1
                print(f"Count is now {count}")
                with open('./errorcounter', 'w') as f:
                    f.write(str(count))

        if dotot:
            remove('./lock')
            remove('./errorcounter')
            remove('./tmpmask')
                
        print(f'{m1} seed {s1}  x  {m2} seed {s2} (ROOT) FINISHED')

    else:
        print(f'{m1} seed {s1}  x  {m2} seed {s2} (RANK {rank}) FINISHED')
