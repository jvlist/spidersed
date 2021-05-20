import pymc3 as pymc
import numpy as np
import pickle
import theano
import theano.tensor as tt
from theano.graph.op import Op
from sed_lib import calc_spec_cxd

#theano.config.exception_verbosity='high'
#theano.config.optimizer='None'

def init(sed_model, data, errors, freqs, freq_pairs, do_sim):

    '''
    Initialize the MCMC model of your choice. 

    Arguments:
    sed_model: name of the model you want. Currently supports only 'dustxcmb'
    data: 1-D array of data points. Make sure data, errors, and freqs are the same order!
    data: 1-D array of error bars or 2-D covariance. Make sure data, errors, and freqs are the same order! Using a full covariance hasn't been tested, but should just work?
    freqs: array of data points. Make sure data, errors, and freqs are the same order!
    freq_pairs: dictionary mapping sqrt(nu1*nu2) frequencies to (nu1, nu2)
    do_sim: Whether this is a simulation run. Sims and data might have different priors.

    Returns:
    model: The (pymc) model object set up with the (sed) model of choice, data, and errors
    '''

    with pymc.Model() as model:
        # Define priors, maybe different for sim and data
        if do_sim:
            a_sync = pymc.Uniform('synchrotron_amplitude', -100, 1000)
            b_sync = pymc.Bound(pymc.Normal, lower=-4, upper=-0.2)('synchrotron_beta', mu=-1.0, sigma=0.37)
            a_dust = pymc.Uniform('dust_amplitude', -100, 5000)
            b_dust = pymc.Uniform('dust_beta', 0.2, 4)
            a_cmb = pymc.Uniform('cmb_amplitude', -4, 4)
            corr_coeff = pymc.Uniform('correlation_coefficient', -0.9, 0.9)
            dxc_corr = pymc.Bound(pymc.Normal, lower=-0.9, upper=0.9)('dustxcmb_correlation', mu=0, sigma=0.073)
        else:
            a_sync = pymc.Uniform('synchrotron_amplitude', -100, 1000)
            b_sync = pymc.Bound(pymc.Normal, lower=-4, upper=-0.2)('synchrotron_beta', mu=-1.0, sigma=0.37)
            a_dust = pymc.Uniform('dust_amplitude', -100, 5000)
            b_dust = pymc.Uniform('dust_beta', 0.2, 4)
            a_cmb = pymc.Uniform('cmb_amplitude', -4, 4)
            corr_coeff = pymc.Uniform('correlation_coefficient', -0.9, 0.9)
            dxc_corr = pymc.Bound(pymc.Normal, lower=-0.9, upper=0.9)('dustxcmb_correlation', mu=0, sigma=0.073)

    def no_model(freq, freq_pairs):
        raise NameError('No model called '+sed_model)

    modeldict = { 'dustxcmb': PowersDxC
                  }
    
    obs = {}

    with model:  # With the model in the context stack, you can add variables just by calling the class
        
        chosen_model = modeldict.get(sed_model, no_model)
        calcs = chosen_model(freqs, freq_pairs)

        powers = calcs(a_sync, b_sync, a_dust, b_dust, a_cmb, corr_coeff, dxc_corr)  
        # Wrapping powers in pymc.Deterministic would save the MCMC trace of the bandpowers, but also slows plotting etc down because that's a lot of extra points (75 bps vs 7 parameters)

        ndim = len(np.shape(errors))
        if ndim == 1:
            dat_cov = np.diag(errors)  # List of error bars -> Diagonal covariance
        elif ndim == 2:
            dat_cov = errors
        else:
            raise ValueError('errors has {} dimensions. Must be 1 or 2'.format(ndim))

        f = pymc.MvNormal('observed_power', mu=powers, cov=dat_cov, observed=data)
        
        return model


### MODELS ###
# Two theano ops per model, one for the model master equation, and one for the gradient. Theano seems to not like something about external fucntions in grad(). The second op wraps it so things work
# Gradients currently seem bugged. Theano errors are gone. Now pymc starts, complains about NUTS initialization failing, then proceeds to show a progress bar for sampling but never makes any progress. For now, using Metropolis-Hastings dodges the issue.

#Dust x CMB model
class PowersDxC(Op):
    def __init__(self, freqs, freq_pairs):
        self.freqs = freqs
        self.pairs = freq_pairs

    itypes = [tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar] #64bit float per param
    otypes = [tt.dvector] #one 75-length float array for bandpowers
    
    def perform(self, node, inputs, outputs):
        As, bs, Ad, bd, Ac, rho, delta = inputs
        vals, _ = calc_spec_cxd(As, bs, Ad, bd, Ac, 0, 0, rho, delta, self.pairs, samples=self.freqs, manual_samples=True)
        outputs[0][0] = np.array(vals)
        #Don't return anything because theano is tricksy and does in-place memory things with the outputs arg
        
    def grad(self, inputs, gouts):
        As, bs, Ad, bd, Ac, rho, delta = inputs
        return [PowersDxCDiff(self.freqs, self.pairs)(As, bs, Ad, bd, Ac, rho, delta)] #There should be chain rule things with gouts here, but it would involve a bunch of math additions to calc_spec and I don't think this ever chains off other variables meaningfully. God help you if that changes.

#Dust x CMB gradient
class PowersDxCDiff(Op):
    def __init__(self, freqs, freq_pairs):
        self.freqs = freqs
        self.pairs = freq_pairs

    itypes = [tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar]
    otypes = [tt.dvector, tt.dvector, tt.dvector, tt.dvector, tt.dvector, tt.dvector, tt.dvector] #75-length float array per param 'cause it's a gradient
    
    def perform(self, node, inputs, outputs):
        As, bs, Ad, bd, Ac, rho, delta = inputs
        gs, _ = calc_spec_cxd(As, bs, Ad, bd, Ac, 0, 0, rho, delta, self.pairs, samples=self.freqs, manual_samples=True, grad=True)
        gs = np.transpose(gs)
        
        grads = np.empty(np.shape(gs)[0], dtype=object) #otypes is specifically a *1-d array-like of 1-d arrays*, NOT a 2-d array. Have to beat numpy with this fact over the head
        for i, g in enumerate(gs):
            grads[i] = g

        outputs[0][0] = grads
      
### END MODELS ###

if __name__ == '__main__':
    print("To use these models, import this file and call fg_model.init")
