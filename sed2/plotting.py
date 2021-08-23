from .base import SED
import pymc3 as pymc
import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
from .fitter import sed_ravel
from .lib import realize_fit_cxd, realize_fit_harm, get_data
import scipy
import json


def plot_posteriors(sed):
    '''
    Plots the posterior distributions of an SED object.

    Arguments:
    sed: An SED object. Posteriors do not have to be computed yet, but it will be much faster if they are.
    '''
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    az.rcParams.update({"plot.max_subplots": 50})
    az.style.use(['arviz-darkgrid', 'arviz-purplish'])

    posts = sed.get_posteriors()

    if sed.sed_model in sed.overell_models:
        etd = ['allell']
    else:
        etd = sed.ells
    
    models = get_data(sed.post_dir, ells_to_do=etd, pols_to_do=[p+'_model' for p in sed.pols])

    for e in etd:
        for p in sed.pols:
            #print(list(posts['allell'].keys()))
            tr = posts[str(e)][p]

            vnames = [n for n in tr.varnames if 'interval__' not in n]# and 'correlation' not in n]

            with models[e][p+'_model']:
                az.plot_pair(tr, kind='kde', divergences=False, 
                             marginals=True, textsize=22, 
                             kde_kwargs = {'contour':True},
                             var_names = vnames
                         )

            fig = plt.gcf()
            fig.suptitle('SED Parameter Posteriors for ell={}, {} Bin'.format(e,p), fontsize=50, y=0.98)
            
            if not os.path.exists('./sed_plots'):
                os.mkdir('./sed_plots')
            
            fname = f'./sed_plots/{e}_{p}_posteriors.png'
            print(f'Saving {fname}')
            plt.savefig(fname)
            plt.close(fig)

    return None


def plot_spectra(sed):
    '''
    Plots the fitted and data spectra of an SED object.

    Arguments:
    sed: An SED object. Posteriors do not have to be computed yet, but it will be much faster if they are.
    '''
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')

    specs = sed.get_spec()
    ac = np.array(sed.map_coll.all_crosses())
    fs = np.array([sed.map_coll.cross_freq(m1,m2) for m1, m2 in ac])
    pairs = {sed.map_coll.cross_freq(m1,m2):(sed.map_coll.freq(m1),sed.map_coll.freq(m2)) for m1, m2 in ac}
    samples = [(sed.map_coll.freq(m1),sed.map_coll.freq(m2)) for m1, m2 in ac]

    args = np.argsort(fs)
    fs = fs[args]
    ac = ac[args]

    posts = sed.get_posteriors()
    
    print(sed.misc_dir+'/'+sed.fl_file)

    blist, slist = sed_ravel(specs, sed.ells, sed.pols, fl=sed.misc_dir+'/'+sed.fl_file, map_coll=sed.map_coll)

    by_ells = []
    dat_ells = []

    for i, (e, p) in enumerate(blist):
        dat, err = slist[i]
    
        if sed.is_overell:
            traces = np.array([posts['allell'][p][param] for param in ['synchrotron_amplitude','synchrotron_beta',
                                                                       'dust_amplitude','dust_beta','cmb_amplitude',
                                                                       'correlation_coefficient', 'dustxcmb_correlation','dust_beta_ell','sync_beta_ell']
                           ])
        else:
            traces = np.array([posts[str(e)][p][param] for param in ['synchrotron_amplitude','synchrotron_beta',
                                                                     'dust_amplitude','dust_beta','cmb_amplitude',
                                                                     'correlation_coefficient', 'dustxcmb_correlation']
                           ])
            
        
        fit_by_model = {'dustxcmb': realize_fit_cxd,
                        'harmonic': realize_fit_harm
                    }

        if sed.is_overell:
            fit_points, fit_errs, _, _ = fit_by_model[sed.sed_model](samples, [e], pairs, traces, component=None, mle=True, around_ml=True)
        else:
            fit_points, fit_errs, _, _ = fit_by_model[sed.sed_model](samples, pairs, traces, component=None, mle=True, around_ml=True)
        

        fit_errs_p, fit_errs_m = map(np.array, zip(*fit_errs))
        dats, daterrs = slist[i]
        res = np.divide( np.subtract(dats, fit_points),  
                         np.sqrt( np.add( np.square(daterrs), np.square( np.mean([fit_errs_p, fit_errs_m], axis=0) 
                                                                     )
                                      )
                              )
                     )



        fig, axes = plt.subplots(2,1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(8,8))
        axes[0].set_yscale('symlog')
        axes[0].errorbar(fs, dat, yerr=err, fmt='+')
        
        axes[0].plot(fs, fit_points, 'k-')
        axes[0].plot(fs, np.add(fit_points,fit_errs_p), 'k--')
        axes[0].plot(fs, np.add(fit_points,-fit_errs_m), 'k--')

        for i, f in enumerate(fs):
            axes[1].plot(f, res[i], '.')#, color=make_color(which_inst[f]))

        ks = scipy.stats.kstest(res, 'norm').pvalue
        axes[1].annotate('KS p = {:.3f}'.format(ks), (.8,.85), xycoords='axes fraction')

        lims = axes[1].get_ylim()
        newlim = max(list(map(abs, lims)) + [1])
        axes[1].set_ylim([-newlim, newlim])

        fig.suptitle('l={} {} Frequency Spectrum'.format(str(e), p))
        axes[0].set_ylabel('D_l')
        axes[1].set_ylabel('Res. (sigma)')
        axes[1].set_xlabel('Freq. (GHz)')
        axes[0].grid(ls='--')
        axes[1].grid(ls='--')

        if not os.path.exists('./sed_plots'):
                os.mkdir('./sed_plots')

        fname = f'./sed_plots/{e}_{p}_spectra.png'
        print(f'Saving {fname}')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(fname)
        plt.close()


        if p == 'EE':
            by_ells.append((e, fit_points[0]))
            dat_ells.append((e, dat[0]))


    es, ps = zip(*by_ells)
    des, dps = zip(*dat_ells)

    ex1 = [1.5*(e/95)**-1.5 for e in es]
    ex2 = [1.5*(e/95)**-3.1 for e in es]

    plt.plot(es, ps, marker='o', c='b', ls='')
    plt.plot(des, dps, marker='o', c='r', ls='') 
    plt.plot(es, ex1, marker='o', c='g', ls='')
    plt.plot(es, ex2, marker='o', c='y', ls='')
    plt.savefig('./sed_plots/byell.png')

    return None


def plot_dustsynch_comp(sed):
    
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')

    ac = np.array(sed.map_coll.all_crosses())
    fs = np.array([sed.map_coll.cross_freq(m1,m2) for m1, m2 in ac])
    pairs = {sed.map_coll.cross_freq(m1,m2):(sed.map_coll.freq(m1),sed.map_coll.freq(m2)) for m1, m2 in ac}

    args = np.argsort(fs)
    fs = fs[args]
    ac = ac[args]

    params = sed.get_posteriors()
    
    dat_90,  confs_90,  dets_90  = {}, {}, {}
    dat_150, confs_150, dets_150 = {}, {}, {}
    dat_c,   confs_c,   dets_c   = {}, {}, {}

    for polcom in ['BB', 'EE']:

        dat_90[polcom],  confs_90[polcom],  dets_90[polcom]  = make_spec_vals(sed, params, (94.5, 94.5), polcom)
        dat_150[polcom], confs_150[polcom], dets_150[polcom] = make_spec_vals(sed, params, (150.5, 150.5), polcom)
        
    sc90 = {'EE':[confs_90['EE'][1],  np.where(dets_90['EE'][0])[0].tolist()], 'BB':[confs_90['BB'][1], np.where(dets_90['BB'][0])[0].tolist()]}
    sc150 = {'EE':[confs_150['EE'][1], np.where(dets_150['EE'][0])[0].tolist()], 'BB':[confs_150['BB'][1], np.where(dets_150['BB'][0])[0].tolist()]}

    dc90 = {'EE':[confs_90['EE'][2],  np.where(dets_90['EE'][1])[0].tolist()], 'BB':[confs_90['BB'][2], np.where(dets_90['BB'][1])[0].tolist()]}
    dc150 = {'EE':[confs_150['EE'][2],  np.where(dets_150['EE'][1])[0].tolist()], 'BB':[confs_150['BB'][2], np.where(dets_150['BB'][1])[0].tolist()]}

    with open('dat_90.json', 'w') as fp:
        json.dump(dat_90, fp)
    with open('sc90.json', 'w') as fp:
        json.dump(sc90, fp)
    with open('dc90.json', 'w') as fp:
        json.dump(dc90, fp)


    return None


def make_spec_vals(sed, params, freq, polcom):

    d_points = []
    d_errs = np.array([[0],[0]])
    d_confs = []
    d_dets = []
    
    s_points = []
    s_errs = np.array([[0],[0]])
    s_confs = []
    s_dets = []

    c_points = []
    c_errs = np.array([[0],[0]])

    r_points = []
    r_errs = np.array([[0],[0]])
    r_confs = []

    rho_points = []
    rho_errs = np.array([[0],[0]])
    rho_confs = []
    

    for ell in map(str, sed.ells):
        
        traces = np.array([params[ell][polcom][p] for p in ['synchrotron_amplitude','synchrotron_beta','dust_amplitude','dust_beta','cmb_amplitude','correlation_coefficient','dustxcmb_correlation']])

        corr_points, corr_errs, corr_confs, corr_detect = realize_fit_cxd([freq], {}, traces, component=['rho'], mle=True, around_ml=True, require_positive=False)
        rho_points += list(corr_points)
        rho_errs = np.hstack((rho_errs, np.array(corr_errs).T))

    
        dust_points, dust_errs, dust_confs, dust_detect = realize_fit_cxd([freq], {}, traces, component=['dust'], mle=True, around_ml=True)
        d_points += list(dust_points)
        d_errs = np.hstack((d_errs, np.array(dust_errs).T))#corr_points+np.array(dust_errs).T+np.array(corr_errs).T))
        d_confs += list(dust_confs)
        d_dets += [not dust_detect[0]]
       
        sync_points, sync_errs, sync_confs, sync_detect = realize_fit_cxd([freq], {}, traces, component=['sync'], mle=True, around_ml=True)
        s_points += list(sync_points)
        s_errs = np.hstack((s_errs, np.array(sync_errs).T))#, corr_points+np.array(sync_errs).T+np.array(corr_errs).T))
        s_confs += list(sync_confs)
        s_dets += [not sync_detect[0]]
        
        cmb_points, cmb_errs, cmb_confs, cmb_detect = realize_fit_cxd([freq], {}, traces, component=['cmb'], mle=True, around_ml=True)
        c_points += list(cmb_points)
        c_errs = np.hstack((c_errs, np.array(cmb_errs).T))
        

        if polcom == 'asdkjl':
            rat_points, rat_errs, rat_confs, rat_detect = realize_fit_cxd([freq], {}, traces, component='ratio', mle=True, around_ml=True)
        else:
            rat_points, rat_errs, rat_confs, rat_detect = realize_fit_cxd([freq], {}, traces, component='ratio', mle=True, around_ml=True)
        r_points += list(rat_points)
        r_errs = np.hstack((r_errs, np.array(rat_errs).T))
        r_confs += list(rat_confs)


    return [(d_points, d_errs.tolist()), (s_points, s_errs.tolist()), (c_points, c_errs.tolist()), (r_points, r_errs.tolist()), (rho_points, rho_errs.tolist())], [r_confs, s_confs, d_confs], [s_dets, d_dets]
