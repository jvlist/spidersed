import pymc3
import numpy as np
import fg_models
import pickle
import sys
from sed_lib import *

def file_write(name, suffix, do_sim):
    t = M.trace(name, chain=None)[:]
    if do_sim:
        with open('/scratch/jvlist/metatraces_xflike_sim/{}.csv'.format(name+suffix), 'a') as f:
            f.write(','+','.join(map(str, t)))
            
    elif not do_sim:
        if excluder != [None]:
            additions = '_'+'_'.join(sorted(excluder))
        else:
            additions = ''
        #with open('/scratch/jvlist/metatraces_xflike'+additions+'/{}.csv'.format(name+suffix), 'a') as f:
        with open('/scratch/jvlist/metatraces_xflike_allneg'+additions+'/{}.csv'.format(name+suffix), 'a') as f:
            f.write(','+','.join(map(str, t)))

def tofl(fls, pol, sbeam=(False, False)):
    fl1, fl2 = fls
    f = np.sqrt(sa.map.spider_fl(fl1, int(ell), int(ell))[pol][0]*sa.map.spider_fl(fl2, int(ell), int(ell))[pol][0])*hp.gauss_beam(np.radians(1), lmax=171)[int(ell)]**2
    #f = hp.gauss_beam(np.radians(1), lmax=171)[int(ell)]**2
    
    fp1 = 1 if fl1 == 150 else 2
    fp2 = 1 if fl2 == 150 else 2

    sb = 1.
    if sbeam[0]:
        sb *= sa.map.spider_beam(fp1, 171)[int(ell)]
    if sbeam[1]:
        sb *= sa.map.spider_beam(fp2, 171)[int(ell)]

    return f*sb

ell_of_interest = float(sys.argv[1])
polcom = str(sys.argv[2])
do_sim = sys.argv[3] == 'True'

pickledir = '/mnt/spider2/jvlist/pickles/'

if do_sim:
    #with open('specs_xflike_test.p', 'r') as pfile:
    #with open(pickledir+'specs_xflike_noiseless.p', 'r') as pfile: 
    with open(pickledir+'specs_xflike_model_sims.p', 'r') as pfile: 
        specs = pickle.load(pfile)
elif not do_sim:
    with open(pickledir+'specs_xflike_data.p', 'r') as pfile: 
    #with open(pickledir+'specs_xflike_data.p', 'r') as pfile: 
        specs = pickle.load(pfile)    

with open(pickledir+'freq_pairs_xflike.p', 'r') as pfile: 
    freq_pairs = pickle.load(pfile)

with open(pickledir+'fl_pairs_xflike.p', 'r') as pfile: 
    fl_pairs = pickle.load(pfile)

#specs = do_all_transfer_mats(specs, fl_pairs)

bin_of_interest = np.argmin(abs(np.array(specs[90.0][0])-ell_of_interest))
ell = specs[90.0][0][bin_of_interest]

fs = np.array(specs.keys(), dtype=float)
sorter = np.argsort(fs)
fs = fs[sorter]
fs_ = np.array(fs)

spiders = []#[94.5, 94.51, 94.52, 94.53, 92.54, 150.5, 150.51, 150.52, 150.53, 150.54]
sbs = {k:(freq_pairs[k][0] in spiders, freq_pairs[k][0] in spiders) for k in freq_pairs.keys()}

ee_flbl2 = [tofl(fl_pairs[f], 2, sbs[f]) for f in fs] 
bb_flbl2 = [tofl(fl_pairs[f], 3, sbs[f]) for f in fs] 
eb_flbl2 = [tofl(fl_pairs[f], 3, sbs[f]) for f in fs] 


unee = lambda spec: np.multiply(np.reciprocal(ee_flbl2), spec)
unbb = lambda spec: np.multiply(np.reciprocal(bb_flbl2), spec)
uneb = lambda spec: np.multiply(np.reciprocal(eb_flbl2), spec)

ee_dls = unee(np.array([specs[freq][1][1][bin_of_interest] for freq in specs], dtype=float)[sorter])
ee_errs = unee(np.array([specs[freq][2][1][bin_of_interest] for freq in specs], dtype=float)[sorter])

bb_dls = unbb(np.array([specs[freq][1][2][bin_of_interest] for freq in specs], dtype=float)[sorter])
bb_errs = unbb(np.array([specs[freq][2][2][bin_of_interest] for freq in specs], dtype=float)[sorter])

eb_dls = uneb(np.array([specs[freq][1][4][bin_of_interest] for freq in specs], dtype=float)[sorter])
eb_errs = uneb(np.array([specs[freq][2][4][bin_of_interest] for freq in specs], dtype=float)[sorter])

if polcom == 'EE':
    foreground_component_model_dustxcmb.init(ee_dls, ee_errs, fs, freq_pairs)
elif polcom == 'BB':
    foreground_component_model_dustxcmb.init(bb_dls, bb_errs, fs, freq_pairs)
elif polcom == 'EB':
    foreground_component_model_dustxcmb.init(eb_dls, eb_errs, fs, freq_pairs)


M = pymc.MCMC(foreground_component_model_dustxcmb)

init_sampler(M)
M.sample(iter=50000, burn=1000, thin=1, progress_bar=False)
print ''

t_a_sync = M.trace('synchrotron_amplitude', chain=None)[:]
t_b_sync = M.trace('synchrotron_beta', chain=None)[:]
t_a_dust = M.trace('dust_amplitude', chain=None)[:]
t_b_dust = M.trace('dust_beta', chain=None)[:]
t_a_cmb = M.trace('cmb_amplitude', chain=None)[:]
t_rho = M.trace('correlation_coefficient', chain=None)[:]
t_delta = M.trace('dustxcmb_correlation', chain=None)[:]

for to_write in ['synchrotron_amplitude','synchrotron_beta','dust_amplitude','dust_beta','cmb_amplitude','correlation_coefficient','dustxcmb_correlation']:
    file_write(to_write, '_'+str(ell_of_interest)+'_'+polcom, do_sim)
