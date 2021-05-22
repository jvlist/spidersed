import numpy as np
import pickle
import sed_lib as sl
import time
import sys
import subprocess
import logging
import logging.handlers
import os.path
import spider_analysis as sa
import json

default_names = { 23: 'map_wmap_k',
                  30: 'map_wmap_ka',
                  40: 'map_wmap_q',
                  60: 'map_wmap_v',
                  90: 'map_wmap_w',
                  94.5: 'map_spider_90',
                  94.51: 'map_spider_90_0', 94.52: 'map_spider_90_1', 94.53: 'map_spider_90_2', 94.54: 'map_spider_90_3',
                  101.3: 'map_planck_100',
                  141.9: 'map_planck_143',
                  150.5: 'map_spider_150a',
                  150.51: 'map_spider_150a_0',150.52: 'map_spider_150a_1',150.53: 'map_spider_150a_2',150.54: 'map_spider_150a_3',
                  220.6: 'map_planck_217',
                  359.7: 'map_planck_353' }

default_noise = { 23: 'map_23_{}',
                  30: 'map_30_{}',
                  40: 'map_40_{}',
                  60: 'map_60_{}',
                  90: 'map_90_{}',
                  94.5: 'map_94_{}',
                  94.51: 'map_94_{}_0', 94.52: 'map_94_{}_1', 94.53: 'map_94_{}_2', 94.54: 'map_94_{}_3',
                  101.3: 'map_100_{}',
                  141.9: 'map_143_{}',
                  150.5: 'map_150_{}',
                  150.51: 'map_150_{}_0', 150.52: 'map_150_{}_1', 150.53: 'map_150_{}_2', 150.54: 'map_150_{}_3',
                  220.6: 'map_217_{}',
                  359.7: 'map_353_{}' }

basefs = [23, 30, 40, 60, 90, 94.5, 101.3, 141.9, 150.5, 220.6, 359.7]
autofs1 = [94.51, 94.52, 94.53, 94.54]
autofs2 = [150.51, 150.52, 150.53, 150.54]
exclfs = [94.5, 150.5]+autofs1+autofs2
f = lambda l: [(f1*f2)**0.5 for f1 in l for f2 in l if (f1*f2)**0.5 not in exclfs] 
default_fs = f(basefs) + f(autofs1) + f(autofs2)


default_suffixes = ('_hm1', '_hm2')


class SEDError(Exception):
    '''Generic SED Error class for exceptions'''

class SED(object):
    '''
    SED object class. Encapsulates configuration and data for a run of the pipeline. 
    This class' methods will generate or read SED pipeline products as appropriate.

    Multiple instances using the same configuration will avoid recomputing things unnecesarily, 
    so you should be able to quickly repopulate a new instance (e.g. in a plotting script) 
    that has the same config as one that's already run with just some disk i/o.

    Attributes:
    ----------
    map_dir 
    map_dict 
    auto_suffixes 
    noise_dict 
    spec_file 
    spec_err 
    post_dir 
    misc_dir 
    do_sim 
    do_cxd 
    fs_to_do 
    sed_model 
    mask 
    ells 
    pols 
    recompute 
    override_config 
    slurm_opts 
    logfile 

    Methods:
    -------
    write_config
    check_config
    get_spec
    get_posteriors
    
    '''
    def __init__(self, 
                 map_dir = '/projects/WCJONES/spider/jvlist/SED_leakage_subtract/',
                 map_dict = default_names,
                 auto_suffixes = default_suffixes,
                 noise_dict = default_noise,
                 spec_file = '/projects/WCJONES/spider/jvlist/specs/SED_leakage_subtract.p', 
                 spec_err_file = '/projects/WCJONES/spider/jvlist/specs/SED_xflike_model_sims.p', 
                 post_dir = '/scratch/gpfs/jvlist/posteriors_leakage_subtract/',
                 misc_dir = '/projects/WCJONES/spider/jvlist/SED_misc/',
                 pairs_file = 'freq_pairs.p',
                 do_sim = False, 
                 do_cxd = True, 
                 fs_to_do = default_fs,
                 sed_model = 'dustxcmb', 
                 mask = np.logical_and(sa.map.standard_point_source_mask(), sa.map.latlon_mask()),
                 ells = [20.0, 45.0, 70.0, 95.0, 120.0, 145.0, 170.0],
                 pols = ['EE', 'BB'],
                 recompute = None,
                 override_config = False,
                 sopts = dict(mem=4, wallt=2,
                              ppn=14, mpi_procs=14, omp_threads=1, nodes=1,
                              scheduler='slurm', nice=None, queue="physics",
                              env_script=os.path.join(os.environ["SPIDER_ROOT"], "spider_py3_setup.sh")),
                 logfile = './sed.log',
             ):
        self.map_dir = map_dir
        self.map_dict = map_dict
        self.auto_suffixes = auto_suffixes
        self.noise_dict = noise_dict
        self.spec_file = spec_file
        self.spec_err_file = spec_err_file
        self.post_dir = post_dir
        self.misc_dir = misc_dir
        self.do_sim = do_sim
        self.do_cxd = do_cxd
        self.sed_model = sed_model
        self.mask = mask
        self.ells = ells
        self.pols = pols
        self.recompute = recompute
        self.override_config = override_config
        self.sopts = sopts
        self.logfile = logfile


        if fs_to_do is None: #If not specified, just take all pairs
            self.fs_to_do = [(f1*f2)**0.5 for f1 in map_dict.keys() for f2 in map_dict.keys()] 
        else:
            self.fs_to_do = fs_to_do

        # Things that will be checked before overwriting posteriors
        self.config_to_check = ['map_dir','do_sim','do_cxd','fs_to_do','sed_model','ells','pols']  

        # Setup logging
        # Change root logger level from WARNING (default) to NOTSET in order for all messages to be delegated.
        logging.getLogger().setLevel(logging.NOTSET)
        
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
        logging.getLogger().addHandler(rotatingHandler)
        
        self.log = logging.getLogger(__name__)
        # End Logging

        # Setup frequency pairs file
        try:
            with open(misc_dir+pairs_file, 'rb') as pfile:
                self.freq_pairs = pickle.load(pfile)
            self.log.info('Found existing frequency pairs file. Loading that.')

        except (IOError, FileNotFoundError):  # If it can't read the file, generate pairs from fs_to_do, then write to file
            self.log.info('Could not read frequency pairs file. Generating pairs.')
            fpfs = self.map_dict.keys()
            self.freq_pairs = { (f1*f2)**0.5:(f1,f2) for f1 in fpfs for f2 in fpfs }
            with open(misc_dir+pairs_file, 'wb') as pfile:
                pickle.dump(self.freq_pairs, pfile)
        # End freq_pairs


    def write_config(self, d):
        with open(d+'/sed.config', 'w') as cf:
            for opt in self.config_to_check:
                optstr = '{} := {}'.format(opt, str(self.__dict__[opt]))
                cf.write(optstr+'\n')
        return None


    def check_config(self, d):
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
                else:
                    self.log.error('%s does not exist in existing config.', opt)
                    self.log.info('Setting override_config=True will overwrite the existing config and proceed.')
                    raise SEDError('Current config conflicts with the existing one, see '+self.logfile)

            else:
                if self.override_config:
                    self.log.warning('%s is %s in existing config, but is now %s. Overriding.', opt, exval, optval)
                else:
                    self.log.error('%s is %s in existing config, but is now %s.', opt, exval, optval)
                    self.log.info('Setting override_config=True will overwrite the existing config and proceed.')
                    raise SEDError('Current config conflicts with the existing one, see '+self.logfile)

        return None


    def get_spec(self, add_noise=None):
        if add_noise is None:
            add_noise = self.do_sim

        fs = self.fs_to_do
        do_recompute = self.recompute in ['spectra']

        if os.path.exists(self.spec_file) and not do_recompute:
            self.log.info('Found existing spectra file at %s. Loading that.', self.spec_file)
            with open(self.spec_file, 'rb') as pfile: 
                return_spec = pickle.load(pfile)

        else:
            specs = {}
            self.log.info('Computing spectra.')
            
            for ftd in self.fs_to_do:
                f1, f2 = self.freq_pairs[ftd]
                self.log.info('%s x %s', f1, f2)

                ext1, ext2 = '', ''  # Clear filename extentions from previous loop
                
                mfile1 = self.map_dir+self.map_dict[f1]
                mfile2 = self.map_dir+self.map_dict[f2]
                
                nfile1 = self.map_dir+self.map_dict[f1]
                nfile2 = self.map_dir+self.map_dict[f2]
                
                if mfile1 == mfile2:
                    ext1 = self.auto_suffixes[0]
                    ext2 = self.auto_suffixes[1]
                    
                ellb, clsb, clse, freq = sl.cross_maps(self.mask, add_noise, mfile1, mfile2, 
                                                       f1, f2, ext1, ext2, 
                                                       noisefile1=nfile1, noisefile2=nfile2
                                                   )
                ellb = [20., 45., 70., 95., 120., 145., 170.]
                specs[freq] = [ellb, clsb, clse]
                    
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


    def get_posteriors(self, do_fl=True):
        
        self.check_config(self.post_dir)
        self.write_config(self.post_dir)

        do_recompute = self.recompute in ['spectra', 'posteriors']
        
        if not do_recompute:
            try:
                etd = list(map(str, self.ells))
                return_params = sl.get_data(self.post_dir, ells_to_do=etd, pols_to_do=self.pols)
                self.log.info('Found existing posteriors in %s. Loading those.', self.post_dir)

            except FileNotFoundError:
                self.log.info('No posterior files found in %s; computing posteriors and saving there',self.post_dir)
                self.make_posteriors(do_fl)

                try:
                    return_params = sl.get_data(self.post_dir, ells_to_do=map(str, self.ells), pols_to_do=self.pols)

                except (FileNotFoundError, IOError, EOFError):
                    self.log.error('Loading posterior files failed after computing posteriors, something probably went wrong in sed_fit.')
                    raise

        else:
            log.info('do_recompute=%s, recomputing posteriors and saving to %s',do_recompute, self.post_dir)
            self.make_posteriors(do_fl)

            try:
                return_params = sl.get_data(self.post_dir, ells_to_do=map(str, self.ells), pols_to_do=self.pols)

            except (IOError, EOFError):
                self.log.error('Loading posterior files failed after computing posteriors, something probably went wrong in sed_fit.')
                raise

        return return_params


    def make_posteriors(self, do_fl):
        
        self.sopts['output'] = './slurm/output_log_sedfit'
        self.sopts['error'] = './slurm/error_log_sedfit'

        elist = json.dumps(self.ells)
        plist = json.dumps(self.pols)

        sa.batch.qsub('python ./sed_fit.py "{}" "{}" "{}" "{}" "{}" "{}" "{}"'.format(
            self.do_sim, self.sed_model, self.spec_file, 
            do_fl, elist, plist, self.post_dir
        ), 
                      name='sedfit', **self.sopts)
                
        self.log.info('Submitted sedfit job')
                
        # Watch slurm, don't proceed until all fit jobs are done.
        while int(subprocess.check_output('squeue | grep sedfit | wc -l', shell=True)) > 0:
            time.sleep(60)

        return None


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
