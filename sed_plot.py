from sed_base import SED
import pymc3 as pymc
import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
from sed_fit import sed_ravel
from sed_lib import realize_fit_cxd
import scipy


def plot_posteriors(sed):
    '''
    Plots the posterior distributions of an SED object.

    Arguments:
    sed: An SED object. Posteriors do not have to be computed yet, but it wil; be fast if they are.
    '''
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    az.style.use(['arviz-darkgrid', 'arviz-purplish'])

    posts = sed.get_posteriors()
    
    for e in sed.ells:
        for p in sed.pols:
            tr = posts[str(e)][p]

            az.plot_pair(tr, kind='kde', divergences=False, 
                         marginals=True, textsize=22, 
                         kde_kwargs = {'contour':True}
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
    sed: An SED object. Posteriors do not have to be computed yet, but it wil; be fast if they are.
    '''
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')

    specs = sed.get_spec()
    ac = np.array(sed.map_coll.all_crosses())
    fs = np.array([sed.map_coll.cross_freq(m1,m2) for m1, m2 in ac])
    pairs = {sed.map_coll.cross_freq(m1,m2):(sed.map_coll.freq(m1),sed.map_coll.freq(m2)) for m1, m2 in ac}

    args = np.argsort(fs)
    fs = fs[args]
    ac = ac[args]

    posts = sed.get_posteriors()

    blist, slist = sed_ravel(specs, sed.ells, sed.pols, fl=sed.misc_dir+'/'+sed.fl_file, map_coll=sed.map_coll)

    for i, (e, p) in enumerate(blist):
        dat, err = slist[i]
    
        traces = np.array([posts[str(e)][p][param] for param in ['synchrotron_amplitude','synchrotron_beta',
                                                   'dust_amplitude','dust_beta','cmb_amplitude',
                                                   'correlation_coefficient', 'dustxcmb_correlation']
                       ])
        
        fit_points, fit_errs, _, _ = realize_fit_cxd(fs, pairs, traces, component=None, mle=True, around_ml=True)
        
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

    return None
