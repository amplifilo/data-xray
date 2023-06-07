import numpy as np
import proplot as pplt
import pandas as pd
import xarray as xr
import nxarray as nxr
import pymc as pm
from pymc import HalfCauchy, Model, Normal, sample
from matplotlib import pyplot as plt
import arviz as az
import pymc.sampling_jax as jx
import jax
from copy import copy
from tqdm.notebook import tqdm
import nxarray
from nexusformat.nexus import *
from scipy.signal import savgol_filter as savgol

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import ARDRegression, LinearRegression, BayesianRidge

from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)


#jax.config.update('jax_platform_name', 'cpu')
#jax.default_backend(), jax.local_device_count()
#print(jax.numpy.ones(3).device()) # TFRT_CPU_0


def pymc_numpyro_glm(x,y):

    """pymc/numpyro fit to 2nd order polynomial. Requires x and y 1D-variables"""

    with Model() as model:  # model specifications in PyMC are wrapped in a with-statement
        # Define priors
        sigma = HalfCauchy("sigma", beta=10)
        intercept = Normal("Intercept", 0, sigma=20)
        poly1 = Normal("poly1", 0, sigma=20)
        poly2 = Normal("poly2", 0, sigma=20)


        # Define likelihood
        likelihood = Normal("y", mu=intercept + poly1 * x + poly2 * x**2, sigma=sigma, observed=y)

        # Inference!
        # draw 3000 posterior samples using NUTS sampling
        #idata = sample(3000)
        idata = jx.sample_numpyro_nuts(3000, progressbar=False, chain_method='parallel')

    kappa_sample = np.array([idata.posterior.poly1[-1] + 2*idata.posterior.poly2[-1]*xx for xx in xdiv])
    _kappa = kappa_sample.mean(axis=-1)
    _kappaz_std = kappa_sample.std(axis=-1)
    
    return [_kappa, _kappa_std]


def pyro_fit(x,y):


    import torch
    import pyro
    import pyro.distributions as dist
    from pyro.infer import SVI, Trace_ELBO, Predictive
    from pyro.optim import Adam
    from pyro.infer.autoguide import AutoDiagonalNormal

    def model(x, y):
        a = pyro.sample("a", dist.Normal(0, 10))
        b = pyro.sample("b", dist.Normal(0, 10))
        c = pyro.sample("c", dist.Normal(0, 10))
        d = pyro.sample("d", dist.Normal(0, 10))
        mean = a*x**3 + b*x**2 + c*x + d
        sigma = pyro.sample("sigma", dist.HalfNormal(10))
        with pyro.plate("data", len(x)):
            pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
    
    # use AutoDiagonalNormal for guide
    guide = AutoDiagonalNormal(model)

    adam_params = {"lr": 0.01}
    optimizer = Adam(adam_params)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    n_steps = 1000
    for step in range(n_steps):
        svi.step(x, y)
        #  if j % 100 == 0:
        #     print(f"[iteration {j+1}] loss: {loss}")

    # extract the fitted parameters and their errors
    param_names = guide.median().keys()
    fitted_params = guide.median()
    fitted_errors = guide.quantiles([0.25, 0.75])

    # Predictive posterior to calculate the confidence interval
    predictive = Predictive(model, guide=guide, num_samples=900)
    samples = predictive(x, y)

    def poly_derivative_fn(x, a, b, c):
        return 3*a*x**2 + 2*b*x + c
    

    poly_derivative_samples = poly_derivative_fn(x[None, :], samples["a"][:, None], samples["b"][:, None], samples["c"][:, None])
    return poly_derivative_samples
    #poly_derivative_mean = poly_derivative_samples.mean(0)
    #poly_derivative_std = poly_derivative_samples.std(0)


def ransac_fit(x,y, deg=2, n_bootstrap=1000):

    x = x[:, np.newaxis]
    
    # Fit a RANSAC regressor
    model = make_pipeline(PolynomialFeatures(deg), RANSACRegressor())
    model.fit(x, y)

    # Extract polynomial coeffs
    # Remember that PolynomialFeatures orders the coefficients in descending order
    # Coeffs are for x^2, x^1 and x^0 terms
    coeff = model.named_steps['ransacregressor'].estimator_.coef_[::-1]

    # Calculate the derivative using the fitted parameters
    poly_der = np.polyder(coeff)
    
    # Bootstrapping for error estimation
    bootstrapped_ders = np.zeros((n_bootstrap, len(x)))

    for i in range(n_bootstrap):
        x_resampled, y_resampled = resample(x, y)
        model.fit(x_resampled, y_resampled)
        _coef = model.named_steps['ransacregressor'].estimator_.coef_[::-1]
        
        bootstrapped_ders[i] = np.polyval(np.polyder(_coef),x).squeeze()

    # Calculate derivatives of bootstrapped models
    #bootstrapped_ders = np.polyder(bootstrapped_coeffs, axis=1)
    
    
    # Derivative values at each x
    _kappa = np.polyval(np.array(poly_der), x)
    _kappa = np.ravel(_kappa)
    
    # Calculate errors for each derivative value
    _kappa_std = np.std(bootstrapped_ders, axis=0)
    _kappa_std = np.ravel(_kappa_std)
    
    return [_kappa, _kappa_std, model]
    

def ard_fit(x,y):
    
    #use ARDRegression to fit and then interpolate the values, return both
    #interpolated and error of the fit

    X = x.reshape((-1, 1))
    
    ard_poly = make_pipeline(
    PolynomialFeatures(degree=4, include_bias=False),
    StandardScaler(),
    ARDRegression(),
    ).fit(X, y)

    #X_interp = np.linspace(min(X),max(X),400).reshape(-1,1)
    _kappa, _kappa_std = ard_poly.predict(X, return_std=True)

    return [_kappa, _kappa_std]


class NCAR_fit(object):
    """ object to fit TAR to a set of IV curves. dset is xarray with a di field"""
    
    def __init__(self, dset, fit=False):
        self.dset = dset
        # if fit:
        #     self.kappa_pymcfit()
        #     self.kappa_pymc_summary()
    
    def kappa_fit(self, func=(lambda x,y: ransac_fit(x,y,deg=2, n_bootstrap=10)), **kwargs):
        
        # defaultKwargs = {
        #     'fit_range':  

        # }

        # kwargs = { **defaultKwargs, **kwargs }
        
        if 'zspec' in kwargs.keys():
            sortindex= np.argsort(kwargs['zspec'])
            di_sorted = np.log10(self.dset.di.data[sortindex]/1e-9)
            xdiv = kwargs['zspec'][sortindex]
        else:
            di_sorted = np.log10(self.dset.di.data/1e-9)
            xdiv = di_sorted[:,0]

        kappaz = []
        kappaz_std = []
        _fitz = []
       
        for _di in tqdm(di_sorted.T):
            
            _di = np.nan_to_num(_di,nan=0.0)
            #assumes that func_fit is func_fit(x,y)
            [_kappa, _kappa_std, model] = func(xdiv,_di)
            _fitz.append(model.predict(xdiv[:,np.newaxis]))
            
           
            #kappa_sample = np.array([idata.posterior.poly1[-1] + 2*idata.posterior.poly2[-1]*xx for xx in xdiv])
            kappaz.append(_kappa)
            kappaz_std.append(_kappa_std)


            # _di = np.nan_to_num(_di,nan=-10.)
            # _kappa, _kappa_std = func(xdiv,_di)
            # print(len(_kappa))
            # kappaz.append(_kappa)
            # kappaz_std.append(kappaz_std)
            
        
        kappaz= np.vstack(kappaz)
        kappaz_std = np.vstack(kappaz_std)

        self.kappa_dict = xr.Dataset(
                data_vars=dict(
                    di_sorted=([ "xdiv","v"], np.array(di_sorted)),
                    kappa=([ "xdiv","v"], np.array(kappaz).T),
                    kappa_std=([ "xdiv","v"], np.array(kappaz_std).T),
                    di_fitted=([ "xdiv","v"], np.array(_fitz).T),
                ),
                coords=dict(
                    v=self.dset.v.data,
                    xdiv=xdiv))

    def kappa_pymcfit(self, **kwargs):
    
        """fit a set of di/dv curves from a TAR dataset. dset is xarray with the di dataset.
        zspec can be passed for a desired z-coordinates"""


        if 'zspec' in kwargs.keys():
            sortindex= np.argsort(kwargs['zspec'])
            di_sorted = np.log10(self.dset.di.data[sortindex]/1e-9)
            xdiv = kwargs['zspec'][sortindex]
        else:
            di_sorted = np.log10(self.dset.di.data/1e-9)
            xdiv = di_sorted[:,0]

        kappaz = []
        kappaz_std = []
        for _di in tqdm(di_sorted.T):

            idata = pymc_numpyro_glm(xdiv,_di)
            kappa_sample = np.array([idata.posterior.poly1[-1] + 2*idata.posterior.poly2[-1]*xx for xx in xdiv])
            kappaz.append(kappa_sample.mean(axis=-1))
            kappaz_std.append(kappa_sample.std(axis=-1))

        # f2,a2 = plt.subplots()
        # a2.plot(di_sorted.T, cycle='viridis')
        self.kappa_dict = xr.Dataset(
            data_vars=dict(
                di_sorted=([ "xdiv","v"], np.array(di_sorted)),
                kappa=([ "xdiv","v"], np.array(kappaz).T),
                kappa_std=([ "xdiv","v"], np.array(kappaz_std).T),
            ),
            coords=dict(
                v=self.dset.v.data,
                xdiv=xdiv))
    
    def kappa_func_fit(self, func=(lambda x,y: ransac_fit(x,y,deg=2, n_bootstrap=10)), **kwargs):
        
        if 'zspec' in kwargs.keys():
            sortindex= np.argsort(kwargs['zspec'])
            di_sorted = np.log10(self.dset.di.data[sortindex]/1e-9)
            xdiv = kwargs['zspec'][sortindex]
        else:
            di_sorted = np.log10(self.dset.di.data/1e-9)
            xdiv = di_sorted[:,0]

        kappaz = []
        kappaz_std = []
        for _di in tqdm(di_sorted.T):
            
            _di = np.nan_to_num(_di,nan=0.0)
            #assumes that func_fit is func_fit(x,y)
            [_kappa, _kappa_std] = func(xdiv,_di)
            
           

            #kappa_sample = np.array([idata.posterior.poly1[-1] + 2*idata.posterior.poly2[-1]*xx for xx in xdiv])
            kappaz.append(_kappa)
            kappaz_std.append(_kappa_std)

        # f2,a2 = plt.subplots()
        # a2.plot(di_sorted.T, cycle='viridis')
        print(np.array(kappaz).shape)
        self.kappa_dict = xr.Dataset(
            data_vars=dict(
                di_sorted=([ "xdiv","v"], np.array(di_sorted)),
                kappa=([ "xdiv","v"], np.array(kappaz).squeeze().T),
                kappa_std=([ "xdiv","v"], np.array(kappaz_std).squeeze().T)
            ),
            coords=dict(
                v=self.dset.v.data,
                xdiv=xdiv))
        
        print("done")
    

    # def kappa_ard_fit(self, **kwargs):


    #     if 'zspec' in kwargs.keys():
    #         sortindex= np.argsort(kwargs['zspec'])
    #         di_sorted = np.log10(self.dset.di.data[sortindex]/1e-9)
    #         xdiv = kwargs['zspec'][sortindex]
    #     else:
    #         di_sorted = np.log10(self.dset.di.data/1e-9)
    #         xdiv = di_sorted[:,0]

    #     kappaz = []
    #     kappaz_std = []
    #     for _di in tqdm(di_sorted.T):
            
    #         _di = np.nan_to_num(_di,nan=0.0)
    #         _afit = ard_fit(xdiv,_di)
            
    #         _kappa = np.gradient(_afit[1])/np.gradient(_afit[0].squeeze())


    #         #kappa_sample = np.array([idata.posterior.poly1[-1] + 2*idata.posterior.poly2[-1]*xx for xx in xdiv])
    #         kappaz.append(np.interp(xdiv,_afit[0].squeeze(),_kappa))
    #         #kappaz_std.append(kappa_sample.std(axis=-1))

    #     # f2,a2 = plt.subplots()
    #     # a2.plot(di_sorted.T, cycle='viridis')
    #     self.kappa_dict = xr.Dataset(
    #         data_vars=dict(
    #             di_sorted=([ "xdiv","v"], np.array(di_sorted)),
    #             kappa=([ "xdiv","v"], np.array(kappaz).T),
    #             #kappa_std=([ "xdiv","v"], np.array(kappaz_std).T),
    #         ),
    #         coords=dict(
    #             v=self.dset.v.data,
    #             xdiv=xdiv))
        
    #     print("done")
        
    
    def kappa_pymc_summary(self, **kwargs):
    
        """present summary of kappa data"""

        #this function plots the kappa values from pre-fitted kappa_dict

        defaultKwargs = {
            'src': None,
            'map_aspect': 2.0,
            'figax': pplt.subplots(refwidth=2.6, refaspect=2.1, nrows=3, sharey=False),
            'zrange':slice(0,-1),
            'kappa_map_ticks':1.,
            'kappa_map_labels':10,
            'voltage_scale':1,
            
        }

        kwargs = { **defaultKwargs, **kwargs }

        # f2,a2 = pplt.subplots()
        # a2.plot(kappa_dict['di_sorted'].T, cycle='viridis')

        src  = [self.kappa_dict if kwargs["src"] is None else kwargs["src"]][0]

        zz = src['kappa'][kwargs["zrange"]]
        xx = src["v"].data*kwargs["voltage_scale"]
        yy = src['xdiv'].data
        _div = src["di_sorted"][kwargs["zrange"]]

        f3, a3 = kwargs['figax']

        a3[0].plot(xx,zz.T,cycle='viridis_r')
        a3[0].format(grid=False, xticklabelsize=12,yticklabelsize=12)
        a3[0].set_ylabel(r"$\it{\kappa / \kappa_{N}}$", labelpad=3.0, size=14)
        a3[0].set_xlabel("bias (mV)", labelpad=0.5,size=14)
        a3[0].set_ylim([0,2])
        a3[0].set_title("")

        mapbounds = [xx.min(),
                     xx.max(),
                     yy.min(),
                     yy.max()
                     ]

        meankappamap = a3[1].imshow(np.flipud(zz), robust=True,cmap="coolwarm",aspect=kwargs["map_aspect"], extent=mapbounds)
        bar = a3[1].colorbar(meankappamap, loc='lr',width=0.7, length=4, ticks=kwargs["kappa_map_ticks"], pad=.5,
                             label=r"$\it{\kappa / \kappa_{N}}$")
        bar.ax.tick_params(labelsize=kwargs["kappa_map_labels"])
        
        a3[1].grid(False)
        #cbar = a3[1].colorbar(meankappamap, loc='lr',width=0.5,length=6,  ticks=.5, pad=.75)
        a3[1].format(xlabel='', ylabel=r"$\it{log(dI/dV)}$", xlabelsize=14,ylabelsize=14,xticklabelsize=12,yticklabelsize=12)

        a3[2].plot(xx, _div.T,cycle='viridis_r')
        a3[2].format(grid=False, xticklabelsize=12,yticklabelsize=12)
        a3[2].set_ylabel(r"$\it{log(dI/dV)}$", labelpad=3.0, size=14)
        a3[2].set_xlabel("bias (mV)", labelpad=0.5,size=14)

    
    def kappa_pymc_plot(self, **kwargs):

        defaultKwargs = {
            'plot_ind': 0,
            'figax': pplt.subplots(nrows=2, refwidth=2.6, refaspect=1.8, sharey=False)
            }

        kwargs = { **defaultKwargs, **kwargs }

        plot_ind = kwargs["plot_ind"]
        f2,a2 = kwargs["figax"]

        _v = self.kappa_dict['v']*1000
        _div = self.kappa_dict['di_sorted'][plot_ind]
        _kappa = self.kappa_dict['kappa'][plot_ind]
        _kappa_err = self.kappa_dict['kappa_std'][plot_ind]

        a2[0].fill_between(_v, _kappa-_kappa_err,_kappa+_kappa_err,color='gray')
        a2[0].format(grid=False, xticklabelsize=12,yticklabelsize=12)
        a2[0].plot(_v, _kappa)
        a2[0].set_ylabel(r"$\it{\kappa / \kappa_{N}}$", labelpad=3.0, size=14)
        a2[0].set_xlabel("bias (mV)", labelpad=0.5,size=14)

        a2[1].plot(_v, _div,color='orange')
        a2[1].format(grid=False, xticklabelsize=12,yticklabelsize=12)
        a2[1].set_ylabel(r"$\it{log(dI/dV)}$", labelpad=3.0, size=14)
        a2[1].set_xlabel("bias (mV)", labelpad=0.5,size=14)

        f2.suptitle(str(plot_ind))

    
    








