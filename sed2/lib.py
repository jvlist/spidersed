#Library for synchrotron/dust fg mcmc
import matplotlib
#matplotlib.use('Agg')
import pymc3
import numpy as np
import matplotlib.pyplot as plt
import pickle
import spider_analysis as sa
import healpy as hp
import scipy.stats
import copy
from numpy import random
import math
from contextlib import contextmanager
import sys

pickledir = '/mnt/spider2/jvlist/pickles/'


def calc_spec_cxd(As, bs, Ad, bd, Ac, start, end, rho, delta, pairs, samples=None, manual_samples=False, component=None, grad=False):
    '''
    Calculate a frequency spectrum with power law dust and synchrotron, and flat CMB. Includes dust x cmb correlation.

    Take first 5 arguments as params, where the spectrum is A_sync*(nu_s/nu_s0)^b_s + A_dust*(nu_d/nu_d0)^b_d + A_cmb.

    component is a list of things to include; 'dust', 'cmb', 'sync', 'rho', 'delta' are allowed. None returns all components.
    '''

    T_dust = 19.6
    T_cmb = 2.7

    pf = 1e20 #unit prefactor, convert to MJy
    '''
    for arg in [As, bs, Ad, bd, Ac, start, end]:
        try:
            arg = float(arg)
        except ValueError:
            print('Type Error: non-numerical value')
            raise
    '''
    if manual_samples:
        pass
    else:
        samples = np.linspace(start, end, 50)

    vals = []
    grads = []
    for f in samples:
        nu_1, nu_2 = pairs[f][0]*1.e9, pairs[f][1]*1.e9

        convert = (pf*planck_bb(nu_1, T_cmb, deriv=True)*pf*planck_bb(nu_2, T_cmb, deriv=True))**-1
        g_1, g_2 = planck_bb(nu_1, T_dust)/planck_bb(353.e9, T_dust), planck_bb(nu_2, T_dust)/planck_bb(353.e9, T_dust)

        cxc = np.sign(Ac)*Ac**2 * pf*planck_bb(nu_1, T_cmb, deriv=True)*pf*planck_bb(nu_2, T_cmb, deriv=True)
        dxd = np.sign(Ad)*Ad**2 * (nu_1*nu_2/353.e9**2)**bd * g_1*g_2
        sxs = np.sign(As)*As**2 * (nu_1*nu_2/23.e9**2)**bs

        sxd = rho*As*Ad *( (nu_1/23.e9)**bs*(nu_2/353.e9)**bd*g_2 \
                           + (nu_2/23.e9)**bs*(nu_1/353.e9)**bd*g_1 )

        cxd = delta*Ac*Ad *( pf*planck_bb(nu_1, T_cmb, deriv=True)*(nu_2/353.e9)**bd*g_2 \
                             + pf*planck_bb(nu_2, T_cmb, deriv=True)*(nu_1/353.e9)**bd*g_1 )
        rat = 0.

        if component != None:
            if 'ratio' in component:
                rat = float(sxs)/(dxd)
                #rat = np.log(float(dxd)/sxs)

            if not 'dust' in component:
                dxd = 0.
            if not 'cmb' in component:
                cxc = 0.
            if not 'sync' in component:
                sxs = 0.
            if not 'rho' in component:
                sxd = 0.
            if not 'delta' in component:
                cxd = 0.

        vals.append(convert*(cxc+dxd+sxs+sxd+cxd)+rat)

        if grad:
            try:
                g_As = 2.*sxs/As + sxd/As
            except (ValueError, ZeroDivisionError):
                g_As = np.nan
            try:
                g_bs = math.log(nu_1*nu_2/23.0e9**2)*sxs \
                       + rho*As*Ad *( math.log(nu_1/23.e9)*(nu_1/23.e9)**bs * (nu_2/353.e9)**bd*g_2 \
                                      + math.log(nu_2/23.e9)*(nu_2/23.e9)**bs * (nu_1/353.e9)**bd*g_1 )
            except (ValueError, ZeroDivisionError):
                g_bs = np.nan
            try:
                g_Ad = 2.*dxd/Ad + sxd/Ad + cxd/Ad
            except (ValueError, ZeroDivisionError):
                g_Ad = np.nan
            try:
                g_bd = math.log(nu_1*nu_2/353.0e9**2)*dxd \
                       + rho*As*Ad *( (nu_1/23.e9)**bs * math.log(nu_2/353.e9)*(nu_2/353.e9)**bd*g_2 \
                                      + (nu_2/23.e9)**bs * math.log(nu_1/353.e9)*(nu_1/353.e9)**bd*g_1 ) \
                       + delta*Ac*Ad *( pf*planck_bb(nu_1, T_cmb, deriv=True) * math.log(nu_2/353.e9)*(nu_2/353.e9)**bd*g_2 \
                                        + pf*planck_bb(nu_2, T_cmb, deriv=True) * math.log(nu_1/353.e9)*(nu_1/353.e9)**bd*g_1 )
            except (ValueError, ZeroDivisionError):
                g_bd = np.nan
            try:
                g_Ac = 2.*cxc/Ac + cxd/Ac 
            except (ValueError, ZeroDivisionError):
                g_Ac = np.nan
            try:
                g_rho = sxd/rho
            except (ValueError, ZeroDivisionError):
                g_rho = np.nan
            try:
                g_delta = cxd/delta
            except (ValueError, ZeroDivisionError):
                g_delta = np.nan

            
            grads.append(np.nan_to_num(convert*np.array([g_As, g_bs, g_Ad, g_bd, g_Ac, g_rho, g_delta])))

    if grad:
        #grads = [np.array([gs[i][j] for i in range(len(gs))]) for j in range(len(gs[0]))] #Theano specifically wants a *list* of 1-d arrays, so np.transpose is no good
        return grads, samples
    else:
        return vals, samples



def make_spectrum(traces, start, end, pairs, samples=None, manual_samples=False, component=None):
    '''
    Make a sim frequency spectrum with power law dust and synchrotron, and flat CMB.

    Take first 5 arguments as [mean, error] lists, where the spectrum is A_sync*(nu_s/nu_s0)^b_s + A_dust*(nu_d/nu_d0)^b_d + A_cmb.
    Error is taken as a fraction of the mean value.
    Last two arguments are start and end frequencies in GHz
    '''

    for arg in [start, end]:
        try:
            arg = float(arg)
        except ValueError:
            print('Type Error: non-numerical values for frequency')
            raise

    r_As = np.random.choice(traces[0])
    r_bs = np.random.choice(traces[1])
    r_Ad = np.random.choice(traces[2])
    r_bd = np.random.choice(traces[3])
    r_Ac = np.random.choice(traces[4])
    r_rho = np.random.choice(traces[5])

    params = [r_As,r_bs,r_Ad,r_bd,r_Ac,r_rho]
    vals, samples = calc_spec(r_As,r_bs,r_Ad,r_bd,r_Ac, start, end, r_rho, pairs, samples=samples, manual_samples=manual_samples, component=component)

    return np.array(vals), np.array(samples), np.array(params)



def make_spectrum_cxd(traces, start, end, pairs, samples=None, manual_samples=False, component=None):
    '''
    Make a sim frequency spectrum with power law dust and synchrotron, and flat CMB. Include CMB/Dust correlation

    Take first 6 arguments as [mean, error] lists, where the spectrum is A_sync*(nu_s/nu_s0)^b_s + A_dust*(nu_d/nu_d0)^b_d + A_cmb.
    Error is taken as a fraction of the mean value.
    Last two arguments are start and end frequencies in GHz
    '''

    for arg in [start, end]:
        try:
            arg = float(arg)
        except ValueError:
            print('Type Error: non-numerical values for frequency')
            raise

    r_As, r_bs, r_Ad, r_bd, r_Ac, r_rho, r_delta = traces

    params = [r_As,r_bs,r_Ad,r_bd,r_Ac,r_rho,r_delta]
    vals, samples = calc_spec_cxd(r_As,r_bs,r_Ad,r_bd,r_Ac, start, end, r_rho, r_delta, pairs, samples=samples, manual_samples=manual_samples, component=component)

    return np.array(vals), np.array(samples), np.array(params)



def make_spectrum_harmonic(traces, start, end, pairs, ell, pol, samples=None, manual_samples=False, component=None):
    '''
    Make a sim frequency spectrum with power law dust and synchrotron, and flat CMB. Include CMB/Dust correlation

    Take first 6 arguments as [mean, error] lists, where the spectrum is A_sync*(nu_s/nu_s0)^b_s + A_dust*(nu_d/nu_d0)^b_d + A_cmb.
    Error is taken as a fraction of the mean value.
    Last two arguments are start and end frequencies in GHz
    '''

    for arg in [start, end]:
        try:
            arg = float(arg)
        except ValueError:
            print('Type Error: non-numerical values for frequency')
            raise

    r_Ad = np.random.choice(traces[0])
    r_bd = np.random.choice(traces[1])
    r_bell = np.random.choice(traces[2])
    r_Ac = np.random.choice(traces[3])
    r_delta = np.random.choice(traces[4])

    params = [r_Ad,r_bd,r_bell,r_Ac,r_delta]
    vals, samples = calc_spec_harmonic(r_Ad, r_bd, r_bell, r_Ac, start, end, r_delta, pairs, ell, pol, samples=samples, manual_samples=manual_samples, component=component)

    return np.array(vals), np.array(samples), np.array(params)


def realize_fit_cxd(samples, pairs, traces, labels=[], bounds={}, view_dists=False, save_dists=False, title_prefix='', 
                    component=None, do_confidence=False, get_shape=False, percent=True, mle=False, around_ml=False, require_positive=True):

    for datname in bounds.keys():
        ind = np.where(np.array(labels) == datname)[0][0]
        keep = np.where(np.logical_and(traces[ind] > bounds[datname][0], traces[ind] < bounds[datname][1]))
        if np.any(keep):
            traces = [d[keep] for d in traces]

    if mle:
        bf_traces = []
        for t in traces:
            k = scipy.stats.gaussian_kde(t)
            x = np.linspace(min(t), max(t), 100)
            bf_traces.append(x[np.argmax(k(x))])

        bf, sams, params = make_spectrum_cxd(bf_traces, 0, 0, pairs, samples=samples, manual_samples=True, component=component)
        '''
        for i in range(len(errs[0,:])):
            dat = errs[:,i]
            k = scipy.stats.gaussian_kde(dat)
            x = np.linspace(min(dat), max(dat), 100)
            bf.append(x[np.argmax(k(x))])
        '''
 
    errs = np.array([])
    weights = np.array([])
    # pick 5000 random sample indices
    ind_draws = random.choice(len(traces[0]), 5000)

    for i in ind_draws:
        vals, sams, params = make_spectrum_cxd(traces[:,i], 0, 0, pairs, samples=samples, manual_samples=True, component=component)

        if errs.any():
            errs = np.vstack((errs, vals))
        else:
            errs = vals

        tw = [traces[0,i], bf_traces[1], traces[2,i], bf_traces[3], traces[4,i], traces[5,i], traces[6,i]]
        w = np.array([weights_by_param(tw, s, pairs, component=component) for s in samples])

        if weights.any():
            weights = np.vstack((weights, w))
        else:
            weights = w


    if get_shape:
        shape = scipy.stats.chi2.fit(errs[:,0], floc=0)

    if view_dists:
        for i in range(len(errs[0])):
            plt.title(title_prefix+str(samples[i]))
            plt.xlabel('D_{l} ')
            plt.ylabel('hits')
            plt.hist(trim_trace(errs[:,i]), bins='fd')
            plt.axvline(x=np.nanmedian(errs[:,i]), c='k', ls='-')
            err_m, err_p = two_sided_std(trim_trace(errs[:,i]), percent=True)
            plt.axvline(x=np.nanmedian(errs[:,i])+err_p, c='k', ls='--')
            plt.axvline(x=np.nanmedian(errs[:,i])-err_m, c='k', ls='--')
            plt.axvline(x=np.percentile(errs, 95, axis=0), c='k', ls=':')

            if get_shape:
                #chi2 = scipy.stats.chi2(shape[0], loc=shape[1], scale=shape[2])
                x = np.linspace(min(errs[:,i]), max(errs[:,i]), num=100)
                #plt.plot(x, chi2.pdf(x), label='chi2 fit')
                #plt.annotate('dof = {:.2f}'.format(shape[0]), (.8,.9), xycoords='axes fraction')

            plt.savefig(title_prefix+str(samples[i])+'.png')
            plt.close()

    if save_dists:
         for i in range(len(errs[0])):
             dist = trim_trace(errs[:,i])
             with open(title_prefix+str(samples[i])+'.p', 'w') as pfile:
                 pickle.dump(dist, pfile)

    if not mle:
        bf = np.nanmedian(errs, axis=0)

    if not around_ml:
        to_return = [bf, np.array([two_sided_std(errs[:,i], percent=percent) for i in range(len(errs[0,:]))])]
        if do_confidence:
            to_return += [np.percentile(errs, 84, axis=0)]
        if get_shape:
            to_return += [shape]

    else:
        perrs, ds = [], []
        for i in range(len(errs[0,:])):
            (m, p), detect = narrowest_percent(errs[:,i], bf[i], fraction=0.68, require_positive=require_positive, weights=weights[:,i])
            perrs.append( (bf[i]-m,p-bf[i]) )
            ds.append(detect)

        to_return = [bf, np.array(perrs), [p[1] for p in perrs], ds]

    return to_return



def realize_fit_harmonic(samples, pairs, traces, ell, pol, labels=[], bounds={}, title_prefix='', component=None):

    for datname in bounds.keys():
        ind = np.where(np.array(labels) == datname)[0][0]
        keep = np.where(np.logical_and(traces[ind] > bounds[datname][0], traces[ind] < bounds[datname][1]))
        if np.any(keep):
            traces = [d[keep] for d in traces]

    errs = np.array([])
    for i in range(1000):
        vals, sams, params = make_spectrum_harmonic(traces, 0, 0, pairs, ell, pol, samples=samples, manual_samples=True, component=component)
        if errs.any():
            errs = np.vstack((errs, vals))
        else:
            errs = vals

    return np.nanmedian(errs, axis=0), np.array([two_sided_std(errs[:,i]) for i in range(len(errs[0,:]))])


def calc_components(samples, pairs, traces):

    dusts = []
    for i in range(5000):

        r_As = np.random.choice(traces[0])
        r_bs = np.random.choice(traces[1])
        r_Ad = np.random.choice(traces[2])
        r_bd = np.random.choice(traces[3])
        r_Ac = np.random.choice(traces[4])
        r_rho = np.random.choice(traces[5])

        if np.any(dusts):
            dusts = np.vstack((dusts,[r_Ad*(f/353)**r_bd for f in samples]))
            syncs = np.vstack((syncs,[r_As*(f/23)**r_bs for f in samples]))
            corrs = np.vstack((corrs,[r_rho*r_As**0.5*r_Ad**0.5*((pairs[f][0]/23)**(r_bs/2)*(pairs[f][1]/353)**(r_bd/2)+(pairs[f][1]/23)**(r_bs/2)*(pairs[f][0]/353)**(r_bd/2)) for f in samples]))
            cmbs = np.vstack((cmbs,[r_Ac for f in samples]))
        else:
            dusts = [r_Ad*(f/353)**r_bd for f in samples]
            syncs = [r_As*(f/23)**r_bs for f in samples]
            corrs = [r_rho*r_As**0.5*r_Ad**0.5*((pairs[f][0]/23)**(r_bs/2)*(pairs[f][1]/353)**(r_bd/2)+(pairs[f][1]/23)**(r_bs/2)*(pairs[f][0]/353)**(r_bd/2)) for f in samples]
            cmbs = [r_Ac for f in samples]

    return np.nanmean(dusts, axis=0), np.nanmean(syncs, axis=0), np.nanmean(corrs, axis=0), np.nanmean(cmbs, axis=0)


def plot_matrix(dats, labels, plot_title, truncate=False, overplot=False, bounds={}, lines={}):

    for datname in bounds.keys():
        ind = np.where(np.array(labels) == datname)[0][0]
        keep = np.where(np.logical_and(dats[ind] > bounds[datname][0], dats[ind] < bounds[datname][1]))
        if np.any(keep):
            dats = [d[keep] for d in dats]

    for i in range(len(dats)):
        if not overplot:
            dats[i] = trim_hist(dats[i])

    if truncate:
        m = np.min([len(j) for j in dats])
        for i in range(len(dats)):
            dats[i] = dats[i][:m]
    else:
        for i in dats:
            for j in dats:
                assert len(i) == len(j), 'Need same data lengths or truncation: {} vs {}'.format(len(i),len(j))



    num = len(dats)

    if not overplot:
        f, axarr = plt.subplots(num, num)
        global plot_matrix_used_bins
        plot_matrix_used_bins = np.empty((num,num), dtype=object)
    else:
        f = plt.figure(plt.get_fignums()[0])
        axarr = np.array(f.axes).reshape((num,num))
    for i in range(len(dats)):
        for j in range(len(dats)):
            if i == j:
                plt.setp(axarr[i,j].get_yticklabels(), visible=False)
                if not overplot:
                    _ , plot_matrix_used_bins[i,j], _ = axarr[i,j].hist(dats[i], bins=50, normed=True)
                else:
                    axarr[i,j].hist(dats[i], bins=plot_matrix_used_bins[i,j], normed=True)
                if labels[i] in lines.keys():
                    refpar = lines[labels[i]]
                    if type(refpar) in [int, float, long]:
                        axarr[i,j].axvline(refpar, color='r')
                    elif type(refpar) == tuple:
                        lim = axarr[i,j].get_xlim()
                        ran = np.linspace(-0.5, 0.5, 100)#(lim[0],lim[1], 100)
                        #try:
                        pdf = getattr(scipy.stats, refpar[0]).pdf(ran, *refpar[1:])
                        axarr[i,j].plot(ran, pdf, color='r')
                        #except AttributeError:
                        #    raise AttributeError('You must use the name of a scipy.stats distribution')
            else:
                if not overplot:
                    histo, xedges, yedges = np.histogram2d(dats[j],dats[i])
                    axarr[j,i].contour(yedges[:-1], xedges[:-1], histo)
                    axarr[j,i].grid()
            if j < i:
                axarr[i,j].set_visible(False)
            if i != j:
                plt.setp(axarr[i,j].get_yticklabels(), visible=False)
                plt.setp(axarr[i,j].get_xticklabels(), visible=False)
            else:
                axarr[i,j].set_xlabel(labels[j])
                axarr[i,j].set_ylabel(labels[i])

    f.subplots_adjust(hspace=0, wspace=0)
    f.set_size_inches(10,10)
    f.suptitle(plot_title)

    return True


def make_color(col_str):
    col_num = 0
    if 'p' in col_str:
        col_num += 16711680
    if 'w' in col_str:
        col_num += 65280
    if 's' in col_str:
        col_num += 255

    s = hex(col_num)[2:].zfill(6)

    return '#' + s


def trim_trace(t):
    cut = 0.05
    rem = int(cut/100.*len(t))
    t = np.array(t)

    low = np.argpartition(t, rem)[:rem]
    high = np.argpartition(t, -rem)[-rem:]
    mask = np.zeros(t.shape, dtype=bool)
    mask[low] = True
    mask[high] = True

    return t[~mask]

def trim_hist(t):
    '''
    drop low population bins from  trace.
    '''
    cut = 5
    t = np.array(t)
    h = np.histogram(t, bins=50)

    small_bins = []
    for i in range(len(h[0])):
        if h[0][i] < (cut/100.)*max(h[0]):
            small_bins.append((h[1][i], h[1][i+1]))

    for b in small_bins:
        t[ np.logical_and(t>=b[0], t<=b[1]) ] = np.nan

    return t[~np.isnan(t)]


def planck_bb(f, T, deriv=False):
    '''
    Planck BB function. f should be in Hz. Deriv=True returns dB/dT.
    '''
    h = 6.626e-34
    c = 2.99e8
    k = 1.38e-23
    if not deriv:
        return (2*h*f**3/c**2)*(1/(2.718**(h*f/(k*T))-1))
    else:
        return (2*h**2*f**4)/(c**2*k*T**2) * np.exp((h*f)/(k*T))/(np.exp((h*f)/(k*T))-1)**2


def get_data(d, ells_to_do=['20.0', '45.0', '70.0', '95.0', '120.0', '145.0', '170.0'], pols_to_do=['EE','BB','EB']):
    traces = {ell:{pol:None for pol in pols_to_do} for ell in ells_to_do}

    for pol in pols_to_do:
        for ell in ells_to_do:
            with open(d+'/{}_{}_trace.p'.format(ell,pol), 'rb') as pfile:
                traces[ell][pol] = pickle.load(pfile)

    return traces


def nanmad(arr):
    return np.nanmedian(np.abs(arr-np.nanmedian(arr)))



def calc_dust_alpha(nu_1, nu_2, bd):
    '''
    Calculate alpha for dust templates between two maps given beta_dust. alpha converts nu_1 -> nu_2
    '''
    nu_1, nu_2  = nu_1*1.e9, nu_2*1.e9
    T_dust = 19.6
    T_cmb = 2.7

    a = lambda beta : (nu_2/nu_1)**beta * planck_bb(nu_2, T_dust)/planck_bb(nu_1, T_dust) * planck_bb(nu_1, T_cmb, deriv=True)/planck_bb(nu_2, T_cmb, deriv=True)

    if type(bd) == list:
        alphas = [a(b) for b in bd]
    elif type(bd) in [int, long, float, complex]:
        alphas = a(bd)
    else:
        raise TypeError('bd must be numeric or list of numerics')

    return alphas



def calc_sync_alpha(nu_1, nu_2, bs):
    '''
    Calculate alpha for sync templates between two maps given beta_dust. alpha converts nu_1 -> nu_2
    '''
    nu_1, nu_2  = nu_1*1.e9, nu_2*1.e9
    T_dust = 19.6
    T_cmb = 2.7

    a = lambda beta : (nu_2/nu_1)**beta * planck_bb(nu_1, T_cmb, deriv=True)/planck_bb(nu_2, T_cmb, deriv=True)

    if type(bs) == list:
        alphas = [a(b) for b in bs]
    elif type(bs) in [int, long, float, complex]:
        alphas = a(bs)
    else:
        raise TypeError('bs must be numeric or list of numerics')

    return alphas



def two_sided_std(arr, percent=False, custom_center=None):
    if custom_center is None:
        m = np.median(arr)
        p = np.partition(arr, len(arr)/2)
    else:
        m = custom_center
        #p = np.concatenate(arr[arr<m]

    if not percent:
        low, high = p[:len(arr)/2], p[len(arr)/2:]
        l = 1.4828*nanmad(np.concatenate((low, 2*m-low)))
        h = 1.4828*nanmad(np.concatenate((high, 2*m-high)))
    else:
        l = m - np.percentile(arr, 16)
        h = np.percentile(arr, 84) - m

    return (l, h)



def get_point_sources(freq):
    ps_dict = {23.0: '/mnt/spider2/jvlist/maps/masks/wmap_polarization_analysis_mask_r9_9yr_v5.fits',
               30.0: '/mnt/spider2/jvlist/maps/masks/wmap_polarization_analysis_mask_r9_9yr_v5.fits',
               40.0: '/mnt/spider2/jvlist/maps/masks/wmap_polarization_analysis_mask_r9_9yr_v5.fits',
               60.0: '/mnt/spider2/jvlist/maps/masks/wmap_polarization_analysis_mask_r9_9yr_v5.fits',
               90.0: '/mnt/spider2/jvlist/maps/masks/wmap_polarization_analysis_mask_r9_9yr_v5.fits',
               94.0: 'p100',
               100.0: 'p100',
               150.0: 'p143',
               143.0: 'p143',
               217.0: 'p217',
               353.0: 'p353'
               }

    freq = float(freq)
    assert freq in ps_dict.keys(), 'Unsupported point source frequency: '+str(freq)

    if ps_dict[freq][0] == '/':
        mask = sa.map.read_map(ps_dict[freq]).astype(bool)
        mask = sa.map.rotate_map(mask, ['G', 'C'])
    elif ps_dict[freq][0] == 'p':
        mask = sa.map.get_point_source_mask(int(ps_dict[freq][1:]), nside=512, radius=0.5)

    return mask



def make_modeled_map(freq, params, ell_bin, seed=203, do_harmonic=False, custom_dust=None, custom_cmb=None, custom_sync=None):

    assert len(params) == 6, 'Need exactly 6 parameters: As, bs, Ad, bd, Ac, rho'
    As, bs, Ad, bd, Ac, rho = params
    freq *= 1.e9

    if not do_harmonic:
        dust_template = (sa.map.read_map('/mnt/spider2/jvlist/maps/inputs/planck_353.fits', field=None)-sa.map.read_map('/mnt/spider2/jvlist/maps/inputs/planck_100.fits', field=None))*1.e6 #K -> uK
    elif ~np.any(custom_dust):
        dust_template = sa.map.read_map('/mnt/spider2/jvlist/maps/harmonic_dust_template.fits', field=None)
        dust_template = sa.map.rotate_map(dust_template, coord=['G','C'])

    sync_template = (sa.map.read_map('/mnt/spider2/jvlist/maps/inputs/wmap_k.fits', field=[0,1,2])-sa.map.read_map('/mnt/spider2/jvlist/maps/inputs/wmap_w.fits', field=[0,1,2]))*1.e3 #mK -> uK
    sync_template = sa.map.rotate_map(sync_template, coord=['G','C'])
    if ~np.any(custom_cmb):
        cmb_template  = 1e6*sa.map.synfast(sa.map.read_cls('r0p03_lensedtotCls.dat'), 512, seed=seed)


    if np.any(custom_dust):
        dust_template = custom_dust
    if np.any(custom_cmb):
        cmb_template = custom_cmb
    if np.any(custom_sync):
        sync_template = custom_sync


    mask = np.logical_and(sa.map.standard_point_source_mask(), sa.map.latlon_mask())

    T_dust = 19.6
    T_cmb = 2.7

    convert_dust = (planck_bb(freq, T_cmb, deriv=True)/(planck_bb(353.0*1e9, T_cmb, deriv=True)))**-1
    convert_sync = (planck_bb(freq, T_cmb, deriv=True)/(planck_bb(23.0*1e9,  T_cmb, deriv=True)))**-1
    pf = 1e20 #unit prefactor, convert to MJy

    g = planck_bb(freq, T_dust)/planck_bb(353.e9, T_dust)

    dust_k = sa.map.estimate_spectrum(dust_template, mask=mask, return_binned=True, lfac=True)[1][1][ell_bin]
    sync_k = sa.map.estimate_spectrum(sync_template, mask=mask, return_binned=True, lfac=True)[1][1][ell_bin]
    cmb_k  = sa.map.estimate_spectrum(cmb_template,  mask=mask, return_binned=True, lfac=True)[1][1][ell_bin]


    dust_scale = Ad/(pf*planck_bb(353.e9, T_cmb, deriv=True)) * dust_k**-.5 * (freq/353.e9)**bd * g * convert_dust
    sync_scale = As/(pf*planck_bb(23.e9, T_cmb, deriv=True)) * sync_k**-.5 * (freq/23.e9)**bs * convert_sync
    cmb_scale  = Ac * cmb_k**-.5

    if not do_harmonic:
        map_to_write = dust_scale*dust_template + sync_scale*sync_template + cmb_scale*cmb_template
    else:
        map_to_write = dust_scale*dust_template + cmb_scale*cmb_template

    return map_to_write



def make_modeled_map_synfast(freq, params, ell_bin=2, ell=70, seed=203,
                             component=None, old_method=False, nside=2048,
                             dust_ref=None, sync_ref=None, cmb_ref=None,
                             return_ref=False):

    assert len(params) == 6, 'Need exactly 6 parameters: As, bs, Ad, bd, Ac, rho'
    As, bs, Ad, bd, Ac, rho = params

    freq *= 1.e9
    pf = 1e20 #unit prefactor, convert to MJy
    T_dust = 19.6
    T_cmb = 2.7

    if old_method:
        dust_template = sa.map.read_map('/mnt/spider2/jvlist/maps/modeled_synfast/dust_sf.fits', field=None)
        sync_template = sa.map.read_map('/mnt/spider2/jvlist/maps/modeled_synfast/sync_sf.fits', field=None)
        cmb_template  = 1e6*sa.map.synfast(sa.map.read_cls('r0p03_lensedtotCls.dat'), 512, seed=seed)
    else:
        if dust_ref is None:
            dust_spec = sa.map.read_cls('dust_power_law.dat')
            dust_ref = 1e6*sa.map.synfast(dust_spec, nside, seed=seed) #K -> uK
        dust_template = dust_ref * (pf*planck_bb(353.e9, T_cmb, deriv=True)) #unit conversion
        if sync_ref is None:
            sync_spec = sa.map.read_cls('sync_power_law.dat')
            sync_ref = 1e6*sa.map.synfast(sync_spec, nside, seed=seed+1000) #K -> uK
        sync_template = sync_ref * (pf*planck_bb(23.e9, T_cmb, deriv=True)) #unit conversion
        if cmb_ref is None:
            cmb_spec = sa.map.read_cls('r0p03_lensedtotCls.dat')
            cmb_ref = 1e6*sa.map.synfast(cmb_spec, nside, seed=seed+10000) #K -> uK
        cmb_template = np.copy(cmb_ref)


    convert_dust = (planck_bb(freq, T_cmb, deriv=True)/(planck_bb(353.0*1e9, T_cmb, deriv=True)))**-1
    convert_sync = (planck_bb(freq, T_cmb, deriv=True)/(planck_bb(23.0*1e9,  T_cmb, deriv=True)))**-1

    g = planck_bb(freq, T_dust)/planck_bb(353.e9, T_dust)
    if old_method:
        mask = np.logical_and(sa.map.standard_point_source_mask(), sa.map.latlon_mask())
        dust_k = sa.map.estimate_spectrum(dust_template, mask=mask, return_binned=True, lfac=True)[1][1][ell_bin]
        sync_k = sa.map.estimate_spectrum(sync_template, mask=mask, return_binned=True, lfac=True)[1][1][ell_bin]
        cmb_k  = sa.map.estimate_spectrum(cmb_template,  mask=mask, return_binned=True, lfac=True)[1][1][ell_bin]


        dust_scale = Ad/(pf*planck_bb(353.e9, T_cmb, deriv=True)) * dust_k**-.5 * (freq/353.e9)**bd * g * convert_dust
        sync_scale = As/(pf*planck_bb(23.e9, T_cmb, deriv=True)) * sync_k**-.5 * (freq/23.e9)**bs * convert_sync
        cmb_scale  = Ac * cmb_k**-.5

    else:
        dust_scale = Ad/(pf*planck_bb(353.e9, T_cmb, deriv=True)) * (freq/353.e9)**bd * g * convert_dust
        sync_scale = As/(pf*planck_bb(23.e9, T_cmb, deriv=True)) * (freq/23.e9)**bs * convert_sync
        cmb_scale  = Ac


    if component != None:
        if 'dust' not in component:
            dust_scale = 0
        if 'sync' not in component:
            sync_scale = 0
        if 'cmb' not in component:
            cmb_scale = 0
    map_to_write = dust_scale*dust_template + sync_scale*sync_template + cmb_scale*cmb_template
    if return_ref:
        return map_to_write, dust_ref, sync_ref, cmb_ref
    else:
        return map_to_write



def seed_corr_coeffs(params, ell_bin, seed, freq_pairs, custom_wmap=False, custom_planck=False):

    assert len(params) == 7, 'Need exactly 7 parameters: As, bs, Ad, bd, Ac, rho, delta'
    As, bs, Ad, bd, Ac, rho, delta = params
    freq = 187.88294228055935*1.e9

    if custom_planck:
        dust_template = (sa.map.read_map(custom_planck, field=None)-sa.map.read_map('/mnt/spider2/jvlist/maps/inputs/planck_100.fits', field=None))*1.e6 #K -> uK
    else:
        dust_template = (sa.map.read_map('/mnt/spider2/jvlist/maps/inputs/planck_353.fits', field=None)-sa.map.read_map('/mnt/spider2/jvlist/maps/inputs/planck_100.fits', field=None))*1.e6 #K -> uK
    if custom_wmap:
        sync_template = (sa.map.read_map(custom_wmap, field=[0,1,2])-sa.map.read_map('/mnt/spider2/jvlist/maps/inputs/wmap_w.fits', field=[0,1,2]))*1.e3 #mK -> uK
    else:
        sync_template = (sa.map.read_map('/mnt/spider2/jvlist/maps/inputs/wmap_k.fits', field=[0,1,2])-sa.map.read_map('/mnt/spider2/jvlist/maps/inputs/wmap_w.fits', field=[0,1,2]))*1.e3 #mK -> uK


    cmb_template  = 1e6*sa.map.synfast(sa.map.read_cls('r0p03_lensedtotCls.dat'), 512, fwhm=np.radians(1.), seed=seed)

    dust_template = sa.map.rotate_map(dust_template, coord=['G','C'])
    sync_template = sa.map.rotate_map(sync_template, coord=['G','C'])
    #cmb_template  = sa.map.rotate_map(cmb_template, coord=['G','C'])

    mask = np.logical_and(sa.map.standard_point_source_mask(), sa.map.latlon_mask())

    T_dust = 19.6
    T_cmb = 2.7



    pf = 1e20 #unit prefactor, convert to MJy

    g = planck_bb(freq, T_dust)/planck_bb(353.e9, T_dust)

    dust_k = sa.map.estimate_spectrum(dust_template, mask=mask, return_binned=True, lfac=True)[1][1][ell_bin]
    sync_k = sa.map.estimate_spectrum(sync_template, mask=mask, return_binned=True, lfac=True)[1][1][ell_bin]
    cmb_k  = sa.map.estimate_spectrum(cmb_template,  mask=mask, return_binned=True, lfac=True)[1][1][ell_bin]


    dust_scale = Ad/(pf*planck_bb(353.e9, T_cmb, deriv=True)) * dust_k**-.5
    sync_scale = As/(pf*planck_bb(23.e9, T_cmb, deriv=True)) * sync_k**-.5
    cmb_scale  = Ac * cmb_k**-.5

    expected_d_cross = calc_spec_cxd(As, bs, Ad, bd, Ac, 0, 0, 1, 1, freq_pairs, samples=[187.88294228055935], manual_samples=True, component=['delta'])[0][0]

    recovered_delta = sa.map.estimate_spectrum(dust_scale*dust_template, map2=cmb_scale*cmb_template, mask=mask, return_binned=True, lfac=True)[1][1][ell_bin]/expected_d_cross


    freq = 90.10549372818508*1.e9

    expected_r_cross = calc_spec_cxd(As, bs, Ad, bd, Ac, 0, 0, 1, 1, freq_pairs, samples=[90.10549372818508], manual_samples=True, component=['rho'])[0][0]

    recovered_rho = sa.map.estimate_spectrum(dust_scale*dust_template, map2=sync_scale*sync_template, mask=mask, return_binned=True, lfac=True)[1][1][ell_bin]/expected_r_cross


    return recovered_rho, recovered_delta



def make_modeled_map_synfast_harmonic(params, seed=203, dl_input=False):

    assert len(params) == 8, 'Need exactly 8 parameters: a_tt, b_tt, a_ee, b_ee, a_bb, b_bb, a_eb, b_eb,'
    a_tt, b_tt, a_ee, b_ee, a_bb, b_bb, a_eb, b_eb = params

    dust_cls = [[a_tt*(0.1/100.)**b_tt]+[a_tt*(ell/100.)**b_tt for ell in range(1,2901)],
                [a_ee*(0.1/100.)**b_ee]+[a_ee*(ell/100.)**b_ee for ell in range(1,2901)],
                [a_bb*(0.1/100.)**b_bb]+[a_bb*(ell/100.)**b_bb for ell in range(1,2901)],
                [a_eb*(0.1/100.)**b_eb]+[a_eb*(ell/100.)**b_eb for ell in range(1,2901)]]

    if dl_input:
        ells = range(2901)
        ells[0] += 1 #this stops div by zero errors and doesn't actually matter, so whatever
        for i in range(4):
            dust_cls[i] = [dust_cls[i][j]/(ells[j]*(ells[j]+1)) for j in range(len(dust_cls[i]))]


    dust_template = 1e6*sa.map.synfast(dust_cls, 512, seed=seed)

    mask = np.logical_and(sa.map.standard_point_source_mask(), sa.map.latlon_mask())

    map_to_write = dust_template

    return map_to_write



def calc_spec_harmonic(Ad, bd, bell, Ac, start, end, delta, pairs, ell, pol, samples=None, manual_samples=False, component=None):
    '''
    Calculate a frequency spectrum with power law dust and synchrotron, and flat CMB. Includes dust x cmb correlation.

    Take first 5 arguments as params, where the spectrum is A_sync*(nu_s/nu_s0)^b_s + A_dust*(nu_d/nu_d0)^b_d + A_cmb.

    component is a list of things to include; 'dust', 'cmb', 'sync', 'rho', 'delta' are allowed. None returns all components.
    '''

    T_dust = 19.6
    T_cmb = 2.7

    pf = 1e20 #unit prefactor, convert to MJy


    with open(pickledir+'binned_cmb.p', 'r') as pfile:
        binned_cmb = pickle.load(pfile)


    cmb_model = np.array([list(binned_cmb[i][0:7]/binned_cmb[i][3]) for i in range(5)])

    cmbkey = {'EE':1, 'BB':2, 'EB':4, 20.:0, 45.:1, 70.:2, 95.:3, 120.:4, 145.:5, 170.5:6}

    for arg in [Ad, bd, Ac, start, end]:
        try:
            arg = float(arg)
        except ValueError:
            print('Type Error: non-numerical value')
            raise

    if manual_samples:
        pass
    else:
        samples = np.linspace(start, end, 50)

    vals = []
    for f in samples:
        nu_1, nu_2 = pairs[f][0]*1.e9, pairs[f][1]*1.e9

        convert = (pf*planck_bb(nu_1, T_cmb, deriv=True)*pf*planck_bb(nu_2, T_cmb, deriv=True))**-1
        g_1, g_2 = planck_bb(nu_1, T_dust)/planck_bb(353.e9, T_dust), planck_bb(nu_2, T_dust)/planck_bb(353.e9, T_dust)

        cxc = Ac**2 * pf*planck_bb(nu_1, T_cmb, deriv=True)*pf*planck_bb(nu_2, T_cmb, deriv=True) * cmb_model[cmbkey[pol]][cmbkey[ell]]
        dxd = (ell/100.)**(bell)*Ad**2 * (nu_1*nu_2/353.e9**2)**bd * g_1*g_2
        cxd = (ell/100.)**(0.5*bell)*delta*Ac*Ad *( pf*planck_bb(nu_1, T_cmb, deriv=True)*(nu_2/353.e9)**bd*g_2 + pf*planck_bb(nu_2, T_cmb, deriv=True)*(nu_1/353.e9)**bd*g_1 )

        if component != None:
            if not 'dust' in component:
                dxd = 0
            if not 'cmb' in component:
                cxc = 0
            if not 'delta' in component:
                cxd = 0

        vals.append(convert*(cxc+dxd+cxd))


    return vals, samples


def readyspec(spec_in, i):
    s = {'EE': spec_in[1][1], 'BB': spec_in[1][2]}
    e = {'EE': spec_in[2][1], 'BB': spec_in[2][2]}

    if i == 0:
        return np.concatenate(([0]*12, s['EE'],[0]*5, s['BB'], [0]*5, [0]*36))
    else:
        return np.concatenate(([0]*12, e['EE'],[0]*5, e['BB'], [0]*5, [0]*36))


def transfer_correct(s, i, f):
    s_r = readyspec(s, i)
    if i == 1:
        s_0 = readyspec(s, 0)
        s_r = np.add(s_r, s_0)
    elif i == 2:
        s_0 = readyspec(s, 0)
        s_r = np.subtract(s_0, s_r)
    mat = sa.nsi.nsi_spectrum.TransferMatrix()
    themat = sa.nsi.nsi_spectrum.TransferMatrix.get(mat, f[0], f[1])
    mul = np.matmul(themat, np.transpose(s_r))
    if i == 0:
        return mul
    if i == 1:
        mul0 = np.matmul(themat, np.transpose(s_0))
        return np.subtract(mul, mul0)
    if i == 2:
        mul0 = np.matmul(themat, np.transpose(s_0))
        return np.subtract(mul0, mul)


def do_transfer(spec, f):
    new = copy.deepcopy(spec)
    d0 = transfer_correct(new, 0, f)
    d1 = transfer_correct(new, 1, f)
    #d2 = transfer_correct(new, 2, f)

    new[1][1] = d0[12:19]
    new[1][2] = d0[24:31]

    new[2][1] = d1[12:19]
    new[2][2] = d1[24:31]

    #new[1][1] = d0[12:19]
    #new[1][2] = d0[24:31]

    return new


def do_all_transfer_mats(specs, fl_pairs):
    new = {}
    for k in specs.keys():
        new[k] = do_transfer(specs[k], fl_pairs[k])

    return new


def dl_to_amp(dl, params, f, pairs, component='dust'):
    '''
    Convert a Dl into an amplitude parameter.
    '''

    As, bs, Ad, bd, Ac, rho, delta = params

    T_dust = 19.6
    T_cmb = 2.7

    pf = 1e20 #unit prefactor, convert to MJy

    nu_1, nu_2 = pairs[f][0]*1.e9, pairs[f][1]*1.e9

    convert = (pf*planck_bb(nu_1, T_cmb, deriv=True)*pf*planck_bb(nu_2, T_cmb, deriv=True))**-1
    g_1, g_2 = planck_bb(nu_1, T_dust)/planck_bb(353.e9, T_dust), planck_bb(nu_2, T_dust)/planck_bb(353.e9, T_dust)

    cxc = Ac**2 * pf*planck_bb(nu_1, T_cmb, deriv=True)*pf*planck_bb(nu_2, T_cmb, deriv=True)
    dxd = Ad**2 * (nu_1*nu_2/353.e9**2)**bd * g_1*g_2
    sxs = As**2 * (nu_1*nu_2/23.e9**2)**bs
    sxd = rho*As*Ad *( (nu_1/23.e9)**bs*(nu_2/353.e9)**bd*g_2 + (nu_2/23.e9)**bs*(nu_1/353.e9)**bd*g_1 )
    cxd = delta*Ac*Ad *( pf*planck_bb(nu_1, T_cmb, deriv=True)*(nu_2/353.e9)**bd*g_2 + pf*planck_bb(nu_2, T_cmb, deriv=True)*(nu_1/353.e9)**bd*g_1 )

    if component == 'dust':
        amp = ( dl * convert**-1 * (nu_1*nu_2/353.e9**2)**(-bd) * (g_1*g_2)**-1 )**0.5
    elif component == 'sync':
        amp = ( dl * convert**-1 * (nu_1*nu_2/23.e9**2)**(-bs) )**0.5
    elif component == 'cmb':
        amp = dl**0.5

    return amp


def narrowest_percent(dat, ml, fraction=0.68, require_positive=True, weights=None):

    detect = True
    weights = np.array(weights)

    if require_positive:
        ps, bins = np.histogram(dat[dat>0], weights=weights[dat>0], bins=200)

        if bins[0] < 0.01*bins[1]: #sometimes assigns some super low value to the bottom bin instead of 0, fix in that case
            bins[0] = 0

        if ml <= 0:
            ml = 0 #if ML value is <0, make it 0
            detect = False
    else:
        ps, bins = np.histogram(dat, bins=200, weights=weights)

    if weights is not None:
        if require_positive:
            ps = ps/np.sum(weights[dat>0])
        else:
            ps = ps/np.sum(weights)
    else:
        if require_positive:
            ps = ps/float(len(dat[dat>0]))
        else:
            ps = ps/float(len(dat))
        
    if require_positive:
        keep = [max(np.searchsorted(bins, ml)-1, 0)] #start with ml bin
    else:
        keep = [np.searchsorted(bins, ml)-1]

    if keep == [200]:
        keep = [199] # Bin edges break keep when ml is very high

    while np.sum(ps[keep]) < fraction:

        l = min(keep) - 1
        h = max(keep) + 1
        if l < 0:
            keep += [h]
        elif h > len(ps) - 1:
            keep += [l]
        elif ps[l] < ps[h]:
            keep += [h]
        else:
            keep += [l]

    top = bins[max(keep) + 1]
    bot = bins[min(keep)]

    if bot == 0 and require_positive:
        detect = False
    elif bot < 0 and top >= 0:
        detect = False

    return (bot, top), detect


def weights_by_param(params, freq, freq_pairs, component=None):
    As, bs, Ad, bd, Ac, rho, delta = params

    T_dust = 19.6
    T_cmb = 2.7
    pf = 1e20 #unit prefactor, convert to MJy

    nu_1, nu_2 = freq_pairs[freq][0]*1.e9, freq_pairs[freq][1]*1.e9
    freq *= 1.e9
    
    convert = (pf*planck_bb(nu_1, T_cmb, deriv=True)*pf*planck_bb(nu_2, T_cmb, deriv=True))**-1
    g_1, g_2 = planck_bb(nu_1, T_dust)/planck_bb(353.e9, T_dust), planck_bb(nu_2, T_dust)/planck_bb(353.e9, T_dust)
    '''
    w_As = np.abs(2*As * (nu_1*nu_2/23.e9**2)**bs)
    w_bs = As**2 * (nu_1*nu_2/23.0e9**2)**bs * math.log(nu_1*nu_2/23.0e9**2)
    w_Ad = np.abs(2*Ad * (nu_1*nu_2/353.e9**2)**bd * g_1*g_2)
    w_bd = Ad**2 * (nu_1*nu_2/353.0e9**2)**bd * math.log(nu_1*nu_2/353.0e9**2)* g_1*g_2
    w_Ac = np.abs(2*Ac)# * pf*planck_bb(nu_1, T_cmb, deriv=True)*pf*planck_bb(nu_2, T_cmb, deriv=True))
    w_rho = 0
    w_delta = 0
    '''
    w_As = np.abs(2*As * (nu_1*nu_2/23.e9**2)**bs)
    w_bs = 0
    w_Ad = np.abs(2*Ad * (nu_1*nu_2/353.e9**2)**bd * g_1*g_2)
    w_bd = 0
    w_Ac = np.abs(2*Ac)# * pf*planck_bb(nu_1, T_cmb, deriv=True)*pf*planck_bb(nu_2, T_cmb, deriv=True))
    w_rho = np.abs(As*Ad) *( (nu_1/23.e9)**bs*(nu_2/353.e9)**bd*g_2 + (nu_2/23.e9)**bs*(nu_1/353.e9)**bd*g_1 )
    w_delta = 0

    if component is not None:
        if not 'dust' in component:
            w_Ad = 0.
            w_bd = 0.
        if not 'cmb' in component:
            w_Ac = 0.
        if not 'sync' in component:
            w_As = 0.
            w_bs = 0.
        if not 'rho' in component:
            r_rho = 0.
        if not 'delta' in component:
            w_delta = 0.
       

    return w_As + w_bs + w_Ad + w_bd + w_Ac + w_rho + w_delta



def cross_maps(mask, add_noise, mapfile1, mapfile2, freq1, freq2, ext1, ext2, noisefile1=None, noisefile2=None):
    clsb_list = []

    m1 = sa.map.tools.read_map(mapfile1+'{}.fits'.format(ext1), field=None)
    m2 = sa.map.tools.read_map(mapfile2+'{}.fits'.format(ext2), field=None)

    if add_noise:
        n1 = sa.map.tools.read_map(noisefile1+'{}.fits'.format(ext1), field=None)
        n2 = sa.map.tools.read_map(noisefile2+'{}.fits'.format(ext2), field=None)

    if add_noise:
        _, clsb, clse = sa.map.tools.estimate_spectrum(m1+n1, map2=m2+n2, lmax=200, coord='C', return_binned=True, lfac=True, return_error=True, mask=mask)
    else:
        _, clsb, clse = sa.map.tools.estimate_spectrum(m1, map2=m2, lmax=200, coord='C', return_binned=True, lfac=True, return_error=True, mask=mask)

    freq = (freq1*freq2)**0.5
    return None, clsb, clse, freq


def grad_wrap(f):
    '''
    An evil function to avoid math errors in calc_spec_cxd gradient option.
    '''
    try:
        return f
    except (ZeroDivisionError, ValueError):
        return None


@contextmanager
def add_prefix(prefix): 
    global is_new_line
    orig_write = sys.stdout.write
    is_new_line = True

    def new_write(*args, **kwargs):
        global is_new_line

        if args[0] == "\n":
            is_new_line = True

        elif is_new_line:
            orig_write("[" + str(prefix) + "]: ")
            is_new_line = False

        orig_write(*args, **kwargs)

    sys.stdout.write = new_write
    yield
    sys.stdout.write = orig_write
