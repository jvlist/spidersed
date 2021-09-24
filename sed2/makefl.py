from sed2.lib import *
from mpi4py import MPI
import sys
import numpy as np
import spider_analysis as sa
import json
import os
import filelock

if __name__ == '__main__':

    _, crosses_to_do, sig_seeds, map_dir, unobs_replace, misc_dir, fl_file, etd = sys.argv

    crosses_to_do = json.loads(crosses_to_do)
    sig_seeds = json.loads(sig_seeds)
    unobs_replace = json.loads(unobs_replace)
    etd = json.loads(etd)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        jobs = [(c, s) for c in crosses_to_do for s in sig_seeds]
        jobs = np.array_split(jobs, size)
    else:
        jobs = None

    jobs = comm.scatter(jobs, root=0)

    results = []

    for [[m1, m2], sig] in jobs:
        unobs1 = sa.map.ud_grade(sa.map.read_map(os.path.join(map_dir, str(sig), m1+'.fits'), field=None), 512)
        unobs2 = sa.map.ud_grade(sa.map.read_map(os.path.join(map_dir, str(sig), m2+'.fits'), field=None), 512)
        reobs1 = sa.map.ud_grade(sa.map.read_map(os.path.join(map_dir, str(sig), m1.replace(*unobs_replace)+'.fits'), field=None), 512)
        reobs2 = sa.map.ud_grade(sa.map.read_map(os.path.join(map_dir, str(sig), m2.replace(*unobs_replace)+'.fits'), field=None), 512)
        
        unspecs = sa.map.estimate_spectrum(unobs1, map2=unobs2, lfac=True, return_binned=True)
        respecs = sa.map.estimate_spectrum(reobs1, map2=reobs2, lfac=True, return_binned=True)
        del unobs1, unobs2, reobs1, reobs2  # Try to conserve memory

        ells = respecs[0]
        fs = np.divide(unspecs[1], respecs[1])

        results.append(((m1,m2), fs))

    allresults = comm.gather(results, root=0)
    
    if rank == 0:
        allresults = [j for i in allresults for j in i]

        lock = filelock.FileLock('./lock', timeout=60)

        if os.path.exists(misc_dir+'/'+fl_file):
            with open(misc_dir+'/'+fl_file, 'rb') as pfile:
                fdict = pickle.load(pfile)
        else:
            fdict = {}

        for c in map(tuple, crosses_to_do):
            rs = [r for (m,r) in allresults if m == c]
            fmeds = np.nanmedian(rs, axis=0)
            fdict[tuple(c)] = [ells, fmeds, fmeds]

        with open(misc_dir+'/'+fl_file, 'wb') as pfile:
            pickle.dump(fdict, pfile)
