import pymc3 as pymc
import numpy as np
import pickle
import theano
import theano.tensor as tt
from theano.graph.op import Op
from sed2.lib import calc_spec_vec, calc_spec_harm

#theano.config.exception_verbosity='high'
#theano.config.optimizer='None'

def init(sed_model, data, errors, ac, map_coll, do_sim, ells=None):
    '''
    Initialize the MCMC model of your choice. 

    Arguments:
    sed_model: name of the model you want. Currently supports only 'dustxcmb'
    data: 1-D array of data points. Make sure data, errors, and freqs are the same order!
    errors: 1-D array of error bars or 2-D covariance. Make sure data, errors, and freqs are the same order! Using a full covariance hasn't been tested, but should just work?
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
            b_sync = pymc.Bound(pymc.Normal, lower=-4, upper=-0.2)('synchrotron_beta', mu=-1.15, sigma=0.37)
            a_dust = pymc.Uniform('dust_amplitude', 100, 5000)
            b_dust = pymc.Uniform('dust_beta', 0.2, 4)
            a_cmb = pymc.Uniform('cmb_amplitude', -4, 4)
            corr_coeff = pymc.Uniform('correlation_coefficient', -0.9, 0.9)
            dxc_corr = pymc.Bound(pymc.Normal, lower=-0.9, upper=0.9)('dustxcmb_correlation', mu=0, sigma=0.073)
        else:
            a_sync = pymc.Uniform('synchrotron_amplitude', -100, 1000)
            b_sync = pymc.Bound(pymc.Normal, lower=-4, upper=-0.2)('synchrotron_beta', mu=-1.15, sigma=0.37)
            a_dust = pymc.Uniform('dust_amplitude', 100, 5000)
            b_dust = pymc.Uniform('dust_beta', 0.2, 4)
            a_cmb = pymc.Uniform('cmb_amplitude', -4, 4, testval=1)
            corr_coeff = pymc.Uniform('correlation_coefficient', -0.9, 0.9)
            dxc_corr = pymc.Bound(pymc.Normal, lower=-0.9, upper=0.9)('dustxcmb_correlation', mu=0, sigma=0.073)

        if sed_model == 'harmonic':
            b_d_ell = pymc.Uniform('dust_beta_ell', -2, 2)
            b_s_ell = pymc.Uniform('sync_beta_ell', -5, 5)
            



    def no_model(freq, freq_pairs):
        raise NameError('No model called '+sed_model)

    modeldict = { 'dustxcmb': PowersDxC,
                  'harmonic': PowersHarm,
                  }
    
    obs = {}

    with model:  # With the model in the context stack, you can add variables just by calling the class
        
        chosen_model = modeldict.get(sed_model, no_model)

        freqs = [(map_coll.freq(m1),map_coll.freq(m2)) for m1, m2 in ac]
        pairs = {map_coll.cross_freq(m1,m2):(map_coll.freq(m1),map_coll.freq(m2)) for m1, m2 in ac}
        calcs = chosen_model(freqs, pairs, ells=ells)

        if sed_model == 'harmonic':
            powers = calcs(a_sync, b_sync, a_dust, b_dust, a_cmb, corr_coeff, dxc_corr, b_d_ell, b_s_ell)
        else:
            powers = calcs(a_sync, b_sync, a_dust, b_dust, a_cmb, corr_coeff, dxc_corr)  
        # Wrapping powers in pymc.Deterministic would save the MCMC trace of the bandpowers, but also slows plotting etc down because that's a lot of extra points (75 bps vs 7 parameters)

        nedim = len(np.shape(errors))
        nddim = len(np.shape(data))

        if nddim == 1:
            if nedim == 1:
                dat_cov = np.diag(errors)  # List of error bars -> Diagonal covariance
            elif nedim == 2:
                dat_cov = errors
            else:
                raise ValueError('errors has {} dimensions. Must be 1 or 2'.format(ndim))

        elif nddim == 2:
            if np.shape(errors) == np.shape(data):
                #powers = tt.flatten(powers)
                data = np.array(data).flatten()
                dat_cov = np.diag(np.array(errors).flatten())  # Errors same shape as data, so flatten, then insert into diagonal covariance
            elif nedim == 3:
                dat_cov = errors
            else:
                raise ValueError('errors has {} dimensions. Must be 2 or 3'.format(ndim))
            
                
        f = pymc.MvNormal('observed_power', mu=powers, cov=dat_cov, observed=data)
        
        return model


### MODELS ###
# Two theano ops per model, one for the model master equation, and one for the gradient. 
# Theano seems to not like something about external fucntions in grad(). The second op wraps it so things work

# Dust x CMB model
class PowersDxC(Op):
    def __init__(self, freqs, pairs, ells=None):
        self.freqs = freqs
        self.pairs = pairs

    itypes = [tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar]  # 64bit float per param
    otypes = [tt.dvector]  # one num-freqs-length float array for bandpowers
    
    def perform(self, node, inputs, outputs):
        vals, _ = calc_spec_vec(inputs, self.freqs)
        outputs[0][0] = np.array(vals) # For some reason you set outputs[0][0] instead of outputs or outputs[0]; I don't get this, but it's what theano wants
        #Don't return anything because theano is tricksy and does in-place memory things with the outputs arg
        
    def grad(self, inputs, gouts):
        J = PowersDxCDiff(self.freqs, self.pairs)(*inputs)  # Get the Jacobian (transpose)

        g = [tt.dot(j, gouts[0]) for j in J]  # Chain rule: Sum over i (dC/df_i*df_i/dx_k)
        return g

# Dust x CMB gradient
class PowersDxCDiff(Op):
    def __init__(self, freqs, freq_pairs, ells=None):
        self.freqs = freqs
        self.pairs = freq_pairs

    itypes = [tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar]
    otypes = [tt.dvector,tt.dvector,tt.dvector,tt.dvector,tt.dvector,tt.dvector,tt.dvector]  # The _Jacobian_ of power wrt input parameter, split into columns
    
    def perform(self, node, inputs, outputs):
        gs, _ = calc_spec_vec(inputs, self.freqs, grad=True)
        
        gs = gs.T  # Transpose before "returning" because you probably cant transpose a list of theano variables.
        for i, g in enumerate(gs):
            outputs[i][0] = g


# Harmonic model
class PowersHarm(Op):
    def __init__(self, freqs, pairs, ells):
        self.freqs = freqs
        self.pairs = pairs
        self.ells = ells

    itypes = [tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar]  # 64bit float per param
    otypes = [tt.dmatrix]  # one num-ell x num-freqs float array for bandpowers
    
    def perform(self, node, inputs, outputs):
        vals, _ = calc_spec_harm(inputs, self.ells, self.freqs)
        outputs[0][0] = np.array(vals) # For some reason you set outputs[0][0] instead of outputs or outputs[0]; I don't get this, but it's what theano wants
        #Don't return anything because theano is tricksy and does in-place memory things with the outputs arg
        
    def grad(self, inputs, gouts):
        J = PowersHarmDiff(self.freqs, self.pairs, self.ells)(*inputs)  # Get the Jacobian (transpose)

        g = [tt.sum(j*gouts[0]) for j in J]  # Chain rule: Sum over i (dC/df_i*df_i/dx_j)
        return g

# Harmonic gradient
class PowersHarmDiff(Op):
    def __init__(self, freqs, freq_pairs, ells):
        self.freqs = freqs
        self.pairs = freq_pairs
        self.ells = ells

    itypes = [tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar]
    otypes = [tt.dmatrix,tt.dmatrix,tt.dmatrix,tt.dmatrix,tt.dmatrix,tt.dmatrix,tt.dmatrix,tt.dmatrix,tt.dmatrix]  # The _Jacobian_ of power wrt input parameter, split into columns
    
    def perform(self, node, inputs, outputs):
        gs, _ = calc_spec_harm(inputs, self.ells, self.freqs, grad=True)
        
        gs = np.transpose(gs, axes=[2,0,1])  # Transpose before "returning" because you probably cant transpose a list of theano variables.
        for i, g in enumerate(gs):
            outputs[i][0] = g
      
### END MODELS ###

if __name__ == '__main__':
    print("To use these models, import this file and call fg_model.init")

