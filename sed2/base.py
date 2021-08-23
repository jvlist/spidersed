import numpy as np
import pickle
import sed2.lib as sl
import time
import sys
import subprocess
import logging
import logging.handlers
import os
import spider_analysis as sa
import json
import healpy as hp
from math import ceil
from scipy.special import comb
import itertools
import math


### Defaults ###

default_names = dict(map_wmap_k        = (23.0, ['a']),
                     map_wmap_ka       = (30.0, ['a']),
                     map_wmap_q        = (40.0, ['a']),
                     map_wmap_v        = (60.0, ['a']),
                     map_wmap_w        = (90.0, ['a']),
                     map_spider_90     = (94.5, ['a', '-auto']),
                     map_spider_90_0   = (94.5, ['b', '-auto']), 
                     map_spider_90_1   = (94.5, ['b', '-auto']),
                     map_spider_90_2   = (94.5, ['b', '-auto']), 
                     map_spider_90_3   = (94.5, ['b', '-auto']),
                     map_planck_100    = (101.3, ['a']),
                     map_planck_143    = (141.9, ['a']),
                     map_spider_150a   = (150.5, ['a', '-auto']),
                     map_spider_150a_0 = (150.5, ['c', '-auto']),
                     map_spider_150a_1 = (150.5, ['c', '-auto']),
                     map_spider_150a_2 = (150.5, ['c', '-auto']),
                     map_spider_150a_3 = (150.5, ['c', '-auto']),
                     map_planck_217    = (220.6, ['a']),
                     map_planck_353    = (359.7, ['a'])
                 )

default_noise = dict(map_wmap_k        = 'map_23_{}',
                     map_wmap_ka       = 'map_30_{}',
                     map_wmap_q        = 'map_40_{}',
                     map_wmap_v        = 'map_60_{}',
                     map_wmap_w        = 'map_90_{}',
                     map_spider_90     = 'map_94_{}',
                     map_spider_90_0   = 'map_94_{}',
                     map_spider_90_1   = 'map_94_{}',
                     map_spider_90_2   = 'map_94_{}',
                     map_spider_90_3   = 'map_94_{}',
                     map_planck_100    = 'map_100_{}',
                     map_planck_143    = 'map_143_{}',
                     map_spider_150a   = 'map_150_{}',
                     map_spider_150a_0 = 'map_150_{}',
                     map_spider_150a_1 = 'map_150_{}',
                     map_spider_150a_2 = 'map_150_{}',
                     map_spider_150a_3 = 'map_150_{}',
                     map_planck_217    = 'map_217_{}',
                     map_planck_353    = 'map_353_{}'
                 )


default_suffixes = ('_hm1', '_hm2')

### End Defaults ###


class MapCollection(dict):
    '''
    A light wrapper for the dict class that eases some common SED operations. Instantiate like dict().

    Entries must be of the form {string : (Number, list of strings)}. The keys are filenames for maps, sans '.fits' and 
    suffixes for auto spectra; values are a tuple or list containing the frequency in the first entry and the grouping 
    list in the second entry. The grouping list determines what maps are crossed according to these rules:
    1. The two maps' lists must share a string which does not start with '-'
    2. The two maps' lists must NOT share a string starting with '-', excluding the special string '-auto'
    3. If a map is paired with itself, its list must not have '-auto'

    Methods:
    freq: Get the frequency of a map in the collection.
    should_cross: Whether two maps in the collection should be crossed.
    cross_freq: Get the geometric mean of the frequencies of two maps in the collection.
    all_crosses: Get all the pairs of maps in the collection which should be crossed.
    '''
    
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

        for k in self:
            # Using ands instead of all means it stops as soon as any line fails which avoids throwing an error here
            well_constructed = type(k) == str \
                               and type(self[k]) in [tuple,list] \
                               and len(self[k]) == 2 \
                               and type(self[k][0]) in [int,float] \
                               and type(self[k][1]) == list \
                               and all([type(j) == str for j in self[k][1]])

            if not well_constructed:
                raise TypeError('MapCollection must have string keys and values which are a (num, list(str)) tuple or list.')

        return None

    def freq(self, m):
        return self[m][0]    
                
    def should_cross(self, m1, m2):
        intersect = set(self[m1][1]) & set(self[m2][1])
        
        b1 = any([not s.startswith('-') for s in intersect])
        b2 = any([s.startswith('-') and s != '-auto' for s in intersect]) and (not m1 == m2)
        b3 = m1 == m2 and '-auto' in intersect

        b = b1 and (not b2) and (not b3)
        return b
        
    def cross_freq(self, m1, m2):
        f1, f2 = self[m1][0], self[m2][0]

        return (f1*f2)**0.5

    def all_crosses(self):
        enum = list(enumerate(self))
        ac = [(m1, m2) for i, m1 in enum for j, m2 in enum if j >= i and self.should_cross(m1, m2)]

        return ac
        

class SEDError(Exception):
    '''Generic SED Error class for exceptions that don't fit something else'''


class SED(object):
    '''
    SED object class. Encapsulates configuration and data for a run of the pipeline. 
    This class' methods will generate or read SED pipeline products as appropriate.

    Multiple instances using the same configuration will avoid recomputing things unnecesarily, 
    so you should be able to quickly repopulate a new instance (e.g. in a plotting script) 
    that has the same config as one that's already run with just some disk i/o.

    Attributes:
    ----------
    -map_dir: string; Directory where the input maps are stored. 
         Default: '/projects/WCJONES/spider/jvlist/SED_leakage_subtract/'
    -map_coll: MapCollection; A MapCollection object defining the map files in map_dir which to be used
         and how they should be crossed. Default: A MapCollection appropriate for the maps use in 
         the bmode paper.
    -auto_suffixes: iterable of strings; An iterable defining the suffixes to be used when taking the
         auto spectrum of maps in the directory. These will be inserted after the filenames in map_coll
         and before '.fits'. Default: ('_hm1', '_hm2')
    -noise_dict: dict; A dictionary mapping filenames used in map_coll to their respective noise maps
         (if noise maps are being added). If filenames have seeds in them, they should be replaced with
         '{}' for iteration. Default: A dict appropriate for maps used in the bmode paper. 
    -seeds: The list of seeds to iterate over when doing a bunch of sim runs. Default: range(20)
    -spec_file: string; Full or relative path filename where spectra will be stored. This will be
         written if it doesn't exist. Default: '/projects/WCJONES/spider/jvlist/specs/SED_leakage_subtract.p'
    -spec_err_file: string; Full or relative path filename where spectra errors are stored. The errors from
         this file will overwrite the ones in spec_file. This is useful for error bars from sims. If None, 
         errors from spec_file will be used. Default: None
    -post_dir: string; Directory where posterior files are/will be stored. If no files are found here, 
         they will be computed. Default: '/scratch/gpfs/jvlist/posteriors_leakage_subtract/'
    -misc_dir: string; Directory used for miscellanous files.
         Default: '/projects/WCJONES/spider/jvlist/SED_misc/'
    -fl_file: string; Filename in misc_dir where fls are/will be stored. If this doesn't exist,
         they will be computed. Default: 'sed_fls.p'
    -do_sim: Bool; Whether this is a sim run. This might affect which priors are used.
         Default: False
    -crosses_to_do: List of tuples; List of map pairs which will be crossed. If None, map_coll.all_crosses
         will be called to populate the list. This is probably best except in specific cases.
         Default: None
    -sed_model: string; Which SED model to use for the MCMC; currently, 'dustxcmb' amd 'harmonic' are supported
         Default: 'dustxcmb'
    -step: string; Step method to use for sampling; must be understandable by pymc. Default: 'metropolis'
    -mask: map object; The mask to use if computing spectra. Default: SPIDER latlon + pointsource mask
    -ells: list of floats; List of ells to do when computing posteriors. These must match bins returned by 
         estimate_spectrum. Default: [20.0, 45.0, 70.0, 95.0, 120.0, 145.0, 170.0]
    -pols: list of strings; List of the polarizations to do when computing posteriors.
         Default: ['EE', 'BB']
    -recompute: List of strings; Which steps of the pipeline to recompute if files are found. Available 
         choices are 'spectra', 'posterior', 'fl'. Everything depending on a given step (eg posteriors
         depend on spectra) will also be recomputed. If None, nothing is recomputed. Default: None
    -override_config: Bool; Before writing new posteriors, the pipeline will check to make sure it isn't
         overwriting differently configured posteriors. Enabling this will force an overwrite and write
         a new config in post_dir. Default: False
    -sopts: dict; slurm options
    -logfile: string; Filename for the log file. Default: './sed.log'
    -other_config: string; An extra confoguration option that will be written/checked. 
         This might be useful, but can often be ignored. Default: ''

    Methods:
    -------
    -write_config: Writes the configuration of the instance to 'sed.config' in post_dir
    -check_config: Checks the confifuration in post_dir and compares to the instance's config. Throws an
         error if it finds a discrepancy.
    -write_fl: Writes the default FlBl2 to fl_file
    -get_spec: Computes or reads spectra 
    -get_posteriors: Computes or reads posteriors
    -make_posteriors: Submits MCMC slurm job for posteriors. This should generally only be used 
         through get_posteriors.
    
    '''
    
    overell_models = ['harmonic']

    def __init__(self, 
                 map_dir = '/projects/WCJONES/spider/jvlist/SED_leakage_subtract/',
                 map_coll = MapCollection(default_names),
                 auto_suffixes = default_suffixes,
                 noise_map = default_noise,
                 sig_seeds = [],#list(map(str, range(203, 214)))
                 seeds = list(map(lambda x: str(x).zfill(4), range(20))),
                 spec_file = '/projects/WCJONES/spider/jvlist/specs/SED_leakage_subtract.p', 
                 spec_err_file = '/projects/WCJONES/spider/jvlist/specs/SED_xflike_model_sims_rekeyed.p', 
                 post_dir = '/scratch/gpfs/jvlist/posteriors_leakage_subtract/',
                 misc_dir = '/projects/WCJONES/spider/jvlist/SED_misc/',
                 fl_file = 'sed_fls.p',
                 do_sim = False, 
                 do_cxd = True, 
                 crosses_to_do = None,
                 sed_model = 'dustxcmb', 
                 step = 'metropolis',
                 mask = np.logical_and(sa.map.standard_point_source_mask(), sa.map.latlon_mask()),
                 ells = [20.0, 45.0, 70.0, 95.0, 120.0, 145.0, 170.0],
                 pols = ['EE', 'BB'],
                 recompute = None,
                 override_config = False,
                 sopts = dict(mem=20, wallt=3, delete=True,
                              ppn=14, mpi_procs=14, nodes=1, srun=True,
                              scheduler='slurm', nice=None, queue="physics",
                              env_script=os.path.join(os.environ["SPIDER_ROOT"], 
                                                      "spider_py3_setup.sh")
                          ),
                 logfile = './sed.log',
                 other_config = '',
             ):
        self.map_dir = map_dir
        self.map_coll = map_coll
        self.auto_suffixes = auto_suffixes
        self.noise_map = noise_map
        self.sig_seeds = sig_seeds
        self.seeds = seeds
        self.spec_file = spec_file
        self.spec_err_file = spec_err_file
        self.post_dir = post_dir
        self.misc_dir = misc_dir
        self.fl_file = fl_file
        self.do_sim = do_sim
        self.sed_model = sed_model
        self.step = step
        self.mask = mask
        self.ells = ells
        self.pols = pols
        self.recompute = recompute
        self.override_config = override_config
        self.sopts = sopts
        self.logfile = logfile
        self.other_config = other_config


        if crosses_to_do is None: #If not specified, just take all pairs
            self.crosses_to_do = map_coll.all_crosses()
        else:
            self.crosses_to_do = crosses_to_do
            
        self.is_overell = self.sed_model in self.overell_models
            

        # Things that will be checked before overwriting posteriors
        self.config_to_check = ['map_dir','fl_file','do_sim','crosses_to_do','sed_model','ells','pols','other_config']  

        # Setup logging
        # Change root logger level from WARNING (default) to NOTSET in order for all messages to be delegated.
        logging.getLogger().setLevel(logging.NOTSET)
        # Clear any handlers to avoid duplicate entries
        if (logging.getLogger().hasHandlers()):
            logging.getLogger().handlers.clear()

        # Add stderr handler, with level WARNING
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        formater = logging.Formatter('%(name)-13s: %(levelname)-8s %(message)s')
        console.setFormatter(formater)
        logging.getLogger().addHandler(console)
        
        # Add file handler, with level INFO
        rotatingHandler = logging.FileHandler(filename=logfile)
        rotatingHandler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        rotatingHandler.setFormatter(formatter)
        logging.getLogger(__name__).addHandler(rotatingHandler)
        
        self.log = logging.getLogger(__name__)
        # End Logging

        return None


    def write_config(self, d):
        '''
        Writes the current instance's config to sed.config in d.

        Arguments:
        -d: string; Directory to write to.
        '''
        with open(d+'/sed.config', 'w') as cf:
            for opt in self.config_to_check:
                optstr = '{} := {}'.format(opt, str(self.__dict__[opt]))
                cf.write(optstr+'\n')
        return None


    def check_config(self, d, safe=False):
        '''
        Checks the current instance's config to against sed.config in d.
        Returns True if it finds a conflict

        Arguments:
        -d: string; Directory to check against.
        -safe; bool; If safe is False, an error will be raised if a config conflict is found.
            Default: False
        '''
        found_conflict = False
        if not os.path.exists(d+'/sed.config'):
            return None  # No existing config means no conflicts, so just proceed

        with open(d+'/sed.config', 'r') as cf:
            existing = dict( [tuple(s.split(' := ')) for s in cf.read().splitlines()] )

        for opt in self.config_to_check:
            optval = str(self.__dict__[opt])
            exval = existing.get(opt, None)
            if exval == optval:
                pass
            elif exval == None:
                if self.override_config:
                    self.log.warning('%s does not exist in existing config. Overriding.', opt)
                    found_conflict = True
                else:
                    self.log.error('%s does not exist in existing config.', opt)
                    self.log.info('Setting override_config=True will overwrite the existing config and proceed.')
                    if not safe:
                        raise SEDError('Current config conflicts with the existing one, see '+self.logfile)
                    found_conflict = True

            else:
                if self.override_config:
                    self.log.warning('%s is %s in existing config, but is now %s. Overriding.', opt, exval, optval)
                    found_conflict = True
                else:
                    self.log.error('%s is %s in existing config, but is now %s.', opt, exval, optval)
                    self.log.info('Setting override_config=True will overwrite the existing config and proceed.')
                    if not safe:
                        raise SEDError('Current config conflicts with the existing one, see '+self.logfile)
                    found_conflict = True

        return found_conflict


    def __check_recompute(self, c):
        if self.recompute is None:
            return False

        else:
            rc = set([self.recompute]) if type(self.recompute) != list else set(self.recompute)
            return bool(rc & set(c))


    def write_fl(self):
        '''
        Writes the default FlBl2 to fl_file. The default is SPIDER's filter transfer function with a 1 deg
        beam. It is assumed all bands use the 150 transfer function except 94.5 and the 90x150 band.
        '''
        if self.fl_file is None:
            self.log.warning('fl_file is None, so write_fl has nothing to do.')
            return None

        already_there = os.path.exists(self.misc_dir+'/'+self.fl_file)
        do_recompute = self.__check_recompute(['fl'])

        if not already_there or do_recompute:
            self.log.info('Writing fl file to %s/%s', self.misc_dir, self.fl_file)
            fdict = {}
            for m1, m2 in self.crosses_to_do:
                f1, f2 = self.map_coll.freq(m1), self.map_coll.freq(m2)
                fl1 = 90 if f1 in [94.5, 94.51, 94.52, 94.53, 94.54] else 150
                fl2 = 90 if f2 in [94.5, 94.51, 94.52, 94.53, 94.54] else 150
                
                etd = list(map(int, self.ells))
                lam = lambda pol: [np.sqrt(sa.map.spider_fl(fl1)[pol][i-8])*
                                   np.sqrt(sa.map.spider_fl(fl2)[pol][i-8])*
                                   hp.gauss_beam(np.radians(1), lmax=700)[i]**2
                                   for i in etd]
                
                farr = [lam(1), lam(2), lam(3), [np.nan]*len(etd), [np.nan]*len(etd), [np.nan]*len(etd)]

                fdict[(m1, m2)] = [etd, farr, farr]

                with open(self.misc_dir+'/'+self.fl_file, 'wb') as pfile:
                    pickle.dump(fdict, pfile)
                
            return None

        else:
            self.log.info('Found existing Fl file, so write_fl has nothing to do.')
            return None


    def get_spec(self, add_noise=None):
        '''
        Checks if spec_file exists and can be read. If it can, it reads that. If it fails (or 
        recompute is appropriately set), it computes new spectra, using the mask and MapCollection.

        Arguments:
        -add_noise: bool; Whether to add noise maps. If None, this defaults to the same as do_sim
            Default: None
        '''
        if add_noise is None:
            add_noise = self.do_sim

        #fs = self.fs_to_do
        do_recompute = self.__check_recompute(['spectra', 'fl'])

        if os.path.exists(self.spec_file) and not do_recompute:
            self.log.info('Found existing spectra file at %s. Loading that.', self.spec_file)
            with open(self.spec_file, 'rb') as pfile: 
                return_spec = pickle.load(pfile)

        else:
            specs = {}
            self.log.info('Computing spectra.')
            
            for m1, m2 in self.crosses_to_do:
                f1, f2 = self.map_coll.freq(m1), self.map_coll.freq(m2)
                self.log.info('%s x %s', m1, m2)

                ext1, ext2 = '', ''  # Clear filename extentions from previous loop
                
                mfile1 = self.map_dir+m1
                mfile2 = self.map_dir+m2
                
                nfile1 = self.map_dir+self.noise_map[m1]
                nfile2 = self.map_dir+self.noise_map[m2]
                
                if mfile1 == mfile2:
                    ext1 = self.auto_suffixes[0]
                    ext2 = self.auto_suffixes[1]
                    
                ellb, clsb, clse, freq = sl.cross_maps(self.mask, add_noise, mfile1, mfile2, 
                                                       f1, f2, ext1, ext2, 
                                                       noisefile1=nfile1, noisefile2=nfile2
                                                   )
                ellb = [20., 45., 70., 95., 120., 145., 170.]
                specs[(m1, m2)] = [ellb, clsb, clse]
                    
            if self.spec_err_file is not None:
                try:
                    with open(self.spec_err_file, 'rb') as pfile:
                        specs_err = pickle.load(pfile)
                except:
                    self.log.error('Failed to load Spectra Error file %s', self.spec_err_file)
                    raise

                for k in specs.keys():
                    specs[k] = [specs[k][0], specs[k][1], specs_err[k][2]]

            return_spec = specs

        self.log.info('Saving computed spectra to %s', self.spec_file)
        with open(self.spec_file, 'wb') as pfile: 
            pickle.dump(return_spec, pfile)
        
        return return_spec


    def get_posteriors(self, fl_file='default'):
        '''
        Checks if there are posteriors in post_dir and can be read. If it can, it reads those. 
        If it fails (or recompute is appropriately set), it submits a slurm job to compute new posteriors
        and then waits for the job to finish.

        Arguments:
        -fl_file: string; File from which to read flbl2. If 'default', it uses self.fl_file
            Default: 'default'
        '''
        if fl_file == 'default':
            fl_file = self.fl_file

        self.fl_file = fl_file  # If this isn't the default, make sure the config checking knows that.
        if fl_file is not None:
            self.write_fl()
        
        found_conflict = self.check_config(self.post_dir)
        self.write_config(self.post_dir)

        do_recompute = self.__check_recompute(['spectra', 'posteriors', 'fl'])

        if self.sed_model in self.overell_models:
            etd = ['allell']
        else:
            etd = list(map(str, self.ells))
        
        if not do_recompute and not found_conflict:
            try:
                return_params = sl.get_data(self.post_dir, ells_to_do=etd, pols_to_do=self.pols)
                self.log.info('Found existing posteriors in %s. Loading those.', self.post_dir)

            except FileNotFoundError:
                self.log.info('No posterior files found in %s; computing posteriors and saving there',self.post_dir)
                self.make_posteriors(fl_file=fl_file)

                try:
                    return_params = sl.get_data(self.post_dir, ells_to_do=etd, pols_to_do=self.pols)

                except (FileNotFoundError, IOError, EOFError):
                    self.log.error('Loading posterior files failed after computing posteriors, something probably went wrong in sed_fit.')
                    raise

        else:
            self.log.info('Recomputing posteriors and saving to %s', self.post_dir)
            self.make_posteriors(fl_file=fl_file)

            try:
                return_params = sl.get_data(self.post_dir, ells_to_do=etd, pols_to_do=self.pols)

            except (IOError, EOFError):
                self.log.error('Loading posterior files failed after computing posteriors, something probably went wrong in sed_fit.')
                raise

        return return_params


    def make_posteriors(self, fl_file=None, manual_slurm=False, step=None):
        if step is None:
            step = self.step
        
        self.sopts['output'] = './slurm/output_log_sedfit'
        self.sopts['error'] = './slurm/error_log_sedfit'

        if self.sed_model in self.overell_models:
            procs = len(self.pols) # One process per pol
        else:
            procs = len(self.ells)*len(self.pols) # One process per ell, pol bin

        nodes = 1 + procs // 40 # One node for every 40 processes
        ppn = ceil(procs / nodes) # Split jobs evenly across nodes

        self.sopts['ppn']=ppn
        self.sopts['mpi_procs']=procs
        self.sopts['nodes']=nodes

        elist = json.dumps(self.ells)
        plist = json.dumps(self.pols)
        mc = json.dumps(self.map_coll)
        if fl_file is not None:
            ff = self.misc_dir+'/'+self.fl_file
        else:
            ff = None

        fpath = os.path.join(os.getenv('SED2_DIR'), 'fitter.py')
            
        sa.batch.qsub("python "+fpath+" '{}' '{}' '{}' '{}' '{}' '{}' '{}' '{}' '{}'".format(
            self.do_sim, self.sed_model, self.spec_file, 
            mc, ff, elist, plist, self.post_dir, step
        ), 
                      name='sedfit', **self.sopts)
                
        startt = time.time()
        self.log.info('Submitted sedfit job')
        print('Submitted sedfit job. Waiting for job to finish... (This takes roughly 1.5 minutes per 1000 samples on della)')
        
        # Watch slurm, don't proceed until fit job is done.
        while int(subprocess.check_output('squeue | grep sedfit | wc -l', shell=True)) > 0:
            time.sleep(60)

        timed = time.time() - startt
        self.log.info('Sampling took {timed} seconds.')

        return None


    def make_fits(self, ell, pol, component=None, fl_file='default'):
        '''
        docstring
        '''
         
        posts = self.get_posteriors(self, fl_file=fl_file)

        found_conflict = self.check_config(self.post_dir)
        self.write_config(self.post_dir)

        do_recompute = self.__check_recompute(['spectra', 'posteriors', 'fl'])

        if self.sed_model in self.overell_models:
            etd = ['allell']
        else:
            etd = list(map(str, self.ells))
        
        posts = self.get_posteriors(self, fl_file=fl_file)[ell][pol]

        traces = np.array([posts[str(e)][p][param] for param in ['synchrotron_amplitude','synchrotron_beta',
                                                   'dust_amplitude','dust_beta','cmb_amplitude',
                                                   'correlation_coefficient', 'dustxcmb_correlation']
                       ])


        fit = sl.realize_fit(self.crosses_to_do, {}, traces, component=component, get_shape=False, percent=True, mle=True, around_ml=True, require_positive=False)

        return fit
        
    def compute_errors(self, fl_file='default', nthreads=40, nnodes=1):
        '''
        docstring
        '''
        if fl_file == 'default':
            fl_file = self.fl_file

        pairs = list(itertools.permutations(self.seeds, 2))

        #procs = nthreads
        #nodes = nnodes
        #ppn = nthreads//nodes

        #self.sopts['ppn'] = ppn
        #self.sopts['mpi_procs'] = procs
        self.sopts['nodes'] = nnodes

        mc = json.dumps(self.map_coll)
        with open('./tmpmask' , 'wb') as pfile:
            pickle.dump(self.mask, pfile)
        #jmask = json.dumps(self.mask.tolist())
        jseeds = json.dumps(list(self.seeds))

        if fl_file is not None:
            ff = self.misc_dir+'/'+self.fl_file
        else:
            ff = None

        fpath = os.path.join(os.getenv('SED2_DIR'), 'simset.py')
            
        if len(self.sig_seeds):
            maxcount = len(self.sig_seeds)*len(self.crosses_to_do)*math.ceil(len(pairs)/nthreads) - 1
        else:
            maxcount = len(self.crosses_to_do)*math.ceil(len(pairs)/nthreads) - 1
            sig_seeds = ['']  # Put one empty string in so the loop does a single iteration

        with open('./errorcounter', 'w') as f:
            f.write(str(maxcount))

        count = 0
        for sig in self.sig_seeds:
            for cross in self.crosses_to_do:
                _pairs = pairs[:]
                while len(_pairs):
                    take = min(nthreads, len(_pairs))
                
                    ppn = take//nnodes
                    self.sopts['ppn'] = ppn
                    self.sopts['mpi_procs'] = take
                    
                    pairlist = _pairs[:take]
                    del _pairs[:take]

                    crosslist = [cross]
                    
                    self.sopts['output'] = f'./slurm/output_log_sedfit_{count}'
                    self.sopts['error'] = f'./slurm/error_log_sedfit_{count}'

                    # Assign each process a place in line for writing results. This prevents race conditions.
                    # It would be nicer to lock files but afaik python doesn't have an elegant way to do this. 
                    playnice = maxcount - count  # playnice is reverse-order to count so that things run in roughly the order they submit

                    if count % 50 == 0:
                        print(f'Submitting job {count}')
                    
                    while int(subprocess.check_output('squeue | grep sederrs | wc -l', shell=True)) > 500:  # Don't submit too many jobs at once to avoid job user limits
                        time.sleep(30)

                    sa.batch.qsub("python "+fpath+" '{}' '{}' '{}' '{}' '{}' '{}' '{}' '{}' '{}' '{}'".format(
                        mc, self.map_dir, ff, jseeds, json.dumps(self.noise_map), json.dumps(self.auto_suffixes),
                        self.spec_err_file, json.dumps(pairlist), json.dumps(crosslist), sig,
                    ), 
                                  name=f'sederrs_{count}', **self.sopts)
                    
                    count += 1

        self.log.info(f'Submitted {count} errors jobs')
        print(f'Submitted {count} errors jobs. This will be a long process; exiting.')
        
        sys.exit()
