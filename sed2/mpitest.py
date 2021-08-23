import spider_analysis as sa
from mpi4py import MPI
import time
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

pix_bytes = np.dtype(np.float64).itemsize
shape = (3, 3145728)

if rank == 0:
    nbytes = np.prod(shape)*pix_bytes
else:
    nbytes = 0

win1 = MPI.Win.Allocate_shared(nbytes, pix_bytes, comm=comm)
win2 = MPI.Win.Allocate_shared(nbytes, pix_bytes, comm=comm)
buf1, itemsize1 = win1.Shared_query(0)
buf2, itemsize2 = win2.Shared_query(0)
map1 = np.ndarray(buffer=buf1, dtype='d', shape=shape)
map2 = np.ndarray(buffer=buf2, dtype='d', shape=shape)

if rank == 0:
    m1 = sa.map.read_map('/projects/WCJONES/spider/jvlist/SED_leakage_subtract/map_planck_100.fits', field=None)
    m2 = sa.map.read_map('/projects/WCJONES/spider/jvlist/SED_leakage_subtract/map_planck_143.fits', field=None)
    print('loaded maps')

    map1[:] = m1
    map2[:] = m2

time.sleep(10)
comm.Barrier()

if rank == 1:
    print('computing spectra')
    spec = sa.map.estimate_spectrum(map1, map2=map2)
    print('done')

comm.Barrier()
