import data_xray.file_io as fio
from data_xray.nanonisio import Spectrum
from data_xray.report import QuickPPT
import numpy as np
import xarray as xr
import pandas as pd
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import proplot as pplt
import re

from pptx import Presentation
from pptx.util import Inches
import functools
import operator
import warnings

warnings.filterwarnings("ignore")


class SinglePointXarray():
    "apply this one to a list of .dat file to compose an xarray"
    def __init__(self,datdict):
        spectra = [self.PackIV(z) for z in datdict];
        zarr = zarr = [np.sum(s[0]) for s in spectra]
        self.zarr = zarr = (zarr - np.min(zarr))
        print([len(s[1]) for s in spectra])
        self.varr = varr = np.mean([s[1] for s in spectra], axis=0)/1e-3
        iarr = np.asarray([s[2] for s in spectra])
        self.iarr = iarr = iarr/1e-9
        self.diarr = diarr = np.asarray([s[3] for s in spectra])



        try:
            log_di = np.log10(np.abs(diarr))
            self.source = xr.Dataset(
            {
            "i": xr.DataArray(data=iarr, dims=["z","v"], coords={"z":zarr, "v":varr}),
            "di": xr.DataArray(data=diarr, dims=["z","v"], coords={"z":zarr, "v":varr}),
            "log_di": xr.DataArray(data=log_di, dims=["z","v"], coords={"z":zarr, "v":varr})
            })

        except:
            print("can't log di for some reason") 
            self.source = xr.Dataset(
            {
            "i": xr.DataArray(data=iarr, dims=["z","v"], coords={"z":zarr, "v":varr}),
            "di": xr.DataArray(data=diarr, dims=["z","v"], coords={"z":zarr, "v":varr})
            })
  
        
    def PackIV(self, fname):
        
        s1 = Spectrum(fname.fname);

        #this here is to check if the data file is a composite with multiple sequential spectra
        if len(np.where(['AVG' in m for m in s1.signals.keys()])[0]):
            z = np.asarray([float(s1.header["Z (m)"]) , float(s1.header["Z offset (m)"])])/1e-9;
            bias = s1.signals["Bias calc (V)"];
            cur = np.mean([s1.signals["Current [AVG] (A)"], s1.signals["Current [AVG] [bwd] (A)"]], axis=0);
        else:
            z = np.asarray([float(s1.header["Z (m)"]) , float(s1.header["Z offset (m)"])])/1e-9;
            cur = np.mean([s1.signals["Current (A)"], s1.signals["Current [bwd] (A)"]], axis=0);
            bias = s1.signals["Bias calc (V)"];
        

        if len(np.where(['LI Demod 1 X [AVG]' in m for m in s1.signals.keys()])[0]):
            lix_inds = np.where(['LI Demod 1 X [AVG]' in m for m in s1.signals.keys()])[0]
            sig1,sig2 = [list(s1.signals.keys())[j] for j in lix_inds]
            diarr = np.mean([s1.signals[sig1], s1.signals[sig2]], axis=0);
        
        elif len(np.where(['LI Demod 1 X' in m for m in s1.signals.keys()])[0]):
            #print(s1.signals.keys())
            # lix_forw = np.where(['LI Demod 1 X (A)' == m for m in s1.signals.keys()])[0]
            # lix_rev = np.where(['LI Demod 1 X (A) [bwd]' == m for m in s1.signals.keys()])[0]
            
            # sig1,sig2 = [list(s1.signals.keys())[j] for j in [lix_forw,lix_rev]]
            diarr = np.mean([s1.signals['LI Demod 1 X (A)'], s1.signals['LI Demod 1 X [bwd] (A)']], axis=0);
        
        else:
            diarr = np.gradient(cur, axis=-1)/np.gradient(bias)
        
        
        
        
        return [z, bias, cur, diarr]



# class SinglePointXarray():
#     "apply this one to a list of .dat file to compose an xarray"
#     def __init__(self,datdict):
#         spectra = [self.PackIV(z) for z in datdict];
#         zarr  = [np.sum(s['z']) for s in spectra]
#         self.zarr = zarr = (zarr - np.min(zarr))
#         self.varr = varr = np.mean([s["bias"] for s in spectra], axis=0)/1e-3

#         for _is, s in enumerate(spectra):
#             try:
#                 s["i"]
#             except:
#                 print("problem with " + str(_is))
#                 return
            
#         iarr = np.asarray([s["i"] for s in spectra])

#         self.iarr = iarr = iarr/1e-9

#         if 'di' in spectra[0].keys():
#             diarr = np.asarray([s['di'] for s in spectra])
#         else:
#             diarr = np.gradient(iarr, axis=1)/np.gradient(varr)

#         self.source = xr.Dataset(
#             {
#             "i": xr.DataArray(data=iarr, dims=["z","v"], coords={"z":zarr, "v":varr}),
#             "di": xr.DataArray(data=diarr, dims=["z","v"], coords={"z":zarr, "v":varr})
#              }
#         )

#     def PackIV(self, fname):

#         def check_record(chan, signalz):

#             chan_bw = re.split('\(',chan)
#             chan_bw = chan_bw[0] + ' [bwd] (' + chan_bw[1]

#             if chan in signalz.keys():
#                 _cdata = signalz[chan]
#                 if chan_bw in signalz.keys():
#                     _cdata = np.mean([_cdata, signalz[chan_bw]], axis=0);
#                 return _cdata
#             else:
#                 return []

#         s1 = Spectrum(fname.fname);
#         _data = {}
#         if "Z (m)" in s1.signals.keys():
#             z = s1.signals["Z (m)"].mean() / 1e-9;
#         else:
#             z = np.asarray([float(s1.header["Z (m)"]) , float(s1.header["Z offset (m)"])])/1e-9;
#             z = np.sum(z)
#         #print(s1.signals.keys())
#         _data['z'] = z
#         keycodes = {"bias":"Bias calc (V)",
#                     "i": "Current (A)",
#                     "di": 'LI Demod 1 X (A)',
#                     "di3": 'LI Demod 3 X (A)',
#                     "diOC": 'OC D1 X (m)'
#                     }

#         for key, code in keycodes.items():
#             _d  = check_record(code, s1.signals)
#             if len(_d):
#                 _data[key] = _d

#         return _data

class DatGridXarray():
    "this is to pack a bunch of dat files, each of which as multiple runs and the [AVG] field"

    def __init__(self, datfiles):
        self.xrs = []
        for gg in datfiles:
            _d = Spectrum(gg)
            self.xrs.append(self.parse_signals(_d))
        self.source = xr.concat(self.xrs, dim="xy")

    def parse_signals(self, spectrum):

        current = []
        m1x = []
        m1y = []
        m3x = []
        m3y = []
        xy = spectrum.header["X (m)"] + ':' + spectrum.header["Y (m)"]

        for k in spectrum.signals.keys():
            if "AVG" in k:
                continue
            else:
                if "Current" in k:
                    current.append(k)
                elif "Demod 3" in k:
                    if "X" in k:
                        m3x.append(k)
                    elif "Y" in k:
                        m3y.append(k)
                elif "Demod 1" in k:
                    if "X" in k:
                        m1x.append(k)
                    elif "Y" in k:
                        m1y.append(k)

        d2r = xr.Dataset(
            data_vars=dict(
                i=(["xy", "z", "bias"], [[spectrum.signals[c] for c in current]]),
                m1x=(["xy", "z", "bias"], [[spectrum.signals[m] for m in m1x]]),
                m1y=(["xy", "z", "bias"], [[spectrum.signals[m] for m in m1y]]),
                m3x=(["xy", "z", "bias"], [[spectrum.signals[m] for m in m3x]]),
                m3y=(["xy", "z", "bias"], [[spectrum.signals[m] for m in m3y]])

            ),
            coords=dict(
                bias=spectrum.signals['Bias calc (V)'],
                z=np.arange(len(m1x)),
                xy=("xy", [xy]))
        )

        return d2r

class AndreevSlopes:

    def __init__(self, source=None, datdict=None):

        if source is None:
            self.source = SinglePointXarray(datdict).source
        else:
            self.source = source

        # if legacy:
        #     spectra = [self.PackIV(z) for z in datdict];

        #     zarr = zarr = [np.sum(s[0]) for s in spectra]

        #     self.zarr = zarr = (zarr - np.min(zarr))
        #     self.varr = varr = np.mean([s[1] for s in spectra], axis=0)/1e-3
        #     iarr = np.asarray([s[-1] for s in spectra])
        #     self.iarr = iarr = iarr/1e-9

        #     diarr = np.gradient(iarr, axis=1)/np.gradient(varr)
        #     try:
        #         log_di = np.log10(np.abs(diarr))

        #     except:
        #         print('error here')
        #         return log_di

        #     self.source = xr.Dataset(
        #         {"i": xr.DataArray(data=iarr, dims=["z","v"], coords={"z":zarr, "v":varr}),
        #         "di": xr.DataArray(data=diarr, dims=["z","v"], coords={"z":zarr, "v":varr}),
        #         "log_di": xr.DataArray(data=log_di, dims=["z","v"], coords={"z":zarr, "v":varr})})

        # else:

    def FitGaussProcess(self, fitChannel="log_di"):
        xx = np.vstack(self.source.v.data)
        self.gprobjects = {}
        gpfits = []
        gpfits_std = []
        gpfits_mse = []

        for j in tqdm(range(len(self.source.z))):
            yy = self.source[fitChannel].isel(z=j)
            gp_kernel = RBF(length_scale_bounds=(0.001, 1.0)) + WhiteKernel(noise_level_bounds=(0.0001, 5.))
            gpr = GaussianProcessRegressor(kernel=gp_kernel)
            gpr.fit(xx, yy)
            gpr_pred, gpr_std = gpr.predict(xx, return_std=True)
            gpfits.append(gpr_pred)
            gpfits_std.append(gpr_std)
            gpfits_mse.append(np.square(np.subtract(yy, gpr_pred)))
            self.gprobjects[j] = gpr

        self.source["fit_di"] = xr.DataArray(data=np.asarray(gpfits), dims=["z", "v"],
                                             coords={"z": self.source.z, "v": self.source.v})
        self.source["fit_di_std"] = xr.DataArray(data=np.asarray(gpfits_std), dims=["z", "v"],
                                                 coords={"z": self.source.z, "v": self.source.v})
        self.source["fit_di_mse"] = xr.DataArray(data=np.asarray(gpfits_mse), dims=["z", "v"],
                                                 coords={"z": self.source.z})

        return self

    def FitAndreevSlopes(self, polynom=3):

        zfits = []
        dzfits = []
        for j in tqdm(self.source.v.data):
            yz = self.source.log_di.sel(v=j, method="nearest")
            yzfit = np.polyfit(x=self.source.z, y=yz, deg=polynom)
            zfits.append(np.polyval(yzfit, self.source.z.data))
            dzfits.append(np.polyval(np.polyder(yzfit), self.source.z.data))

        self.source["fit_dlogi_dz"] = xr.DataArray(np.asarray(zfits).T, dims=["z", "v"],
                                                   coords={"z": self.source.z, "v": self.source.v})
        self.source["slopes"] = xr.DataArray(np.asarray(dzfits).T, dims=["z", "v"],
                                             coords={"z": self.source.z, "v": self.source.v})
        return self

    def PackIV(self, fname):

        s1 = Spectrum(fname.fname);
        z = np.asarray([float(s1.header["Z (m)"]), float(s1.header["Z offset (m)"])]) / 1e-9;
        cur = np.mean([s1.signals["Current (A)"], s1.signals["Current [bwd] (A)"]], axis=0);
        bias = s1.signals["Bias calc (V)"];
        return [z, bias, cur]

    @staticmethod
    def FftDerivative(y, x, thresh=None):
        from scipy.fftpack import fft, ifft, fftshift, fftfreq

        y_fftfreq = fftfreq(y.size, 1 / len(x))
        y_fft = fft(y)
        if thresh != None:
            y_fft[(np.abs(y_fftfreq) > thresh)] = 0

        return ifft(1j * y_fftfreq * y_fft)

    @staticmethod
    def other_derivatives(y, x):
        from derivative import dxdt

        t = np.linspace(0, 2 * np.pi, 50)
        x = np.sin(t)

        # 1. Finite differences with central differencing using 3 points.
        result1 = dxdt(x, t, kind="finite_difference", k=1)

        # 2. Savitzky-Golay using cubic polynomials to fit in a centered window of length 1
        result2 = dxdt(x, t, kind="savitzky_golay", left=.5, right=.5, order=3)

        # 3. Spectral derivative
        result3 = dxdt(x, t, kind="spectral")

        # 4. Spline derivative with smoothing set to 0.01
        result4 = dxdt(x, t, kind="spline", s=1e-2)

        # 5. Total variational derivative with regularization set to 0.01
        result5 = dxdt(x, t, kind="trend_filtered", order=0, alpha=1e-2)

class AndreevPlots:
    def __init__(self, datdict):
        self.src = AndreevSlopes(datdict);

    @staticmethod
    def PlotPlot(*args, **kwargs):
        f2, a2 = plt.subplots(1, 1);
        a2.plot(*args, **kwargs);
        return f2

class AndreeSlopesGrid:
    def __init__(self):
        pass

    def FitGaussProcess(self, ivz2):
        from copy import deepcopy
        ivz = deepcopy(ivz2)

        xx = ivz.v.values.reshape(-1, 1)
        gprobjects = {}
        gpfits = []
        gpfits_std = []
        gpfits_mse = []
        f22, a22 = plt.subplots(1, 1)
        for jz in tqdm(ivz.z):
            for jx in (ivz.x):
                for jy in (ivz.y):
                    yy = ivz.sel(z=jz, x=jx, y=jy).values.reshape(-1, 1)
                    yy = np.nan_to_num(yy, 0)
                    gp_kernel = RBF() + WhiteKernel() + ConstantKernel()  # noise_level_bounds=(0.0001,5.); length_scale_bounds=(0.001,1.0)

                    gpr = GaussianProcessRegressor(kernel=gp_kernel)
                    gpr.fit(xx, yy)

                    gpr_pred, gpr_std = gpr.predict(xx, return_std=True)
                    a22.plot(xx, yy, 'b.')
                    a22.plot(xx, gpr_pred, 'r--')

                    ivz.loc[dict(z=jz, x=jx, y=jy)] = np.ravel(gpr_pred)
                    # gpfits.append(gpr_pred)
                    # gpfits_std.append(gpr_std)
                    # gpfits_mse.append(np.square(np.subtract(yy,gpr_pred)))
                    # gprobjects[j] = gpr

        return ivz

class Ncar_Set(object):
    
    def __init__(self, _set, ref = 2, init = 4, sym = False):
        self.source = _set
        self.div = self.source.di.data*1e10
        if sym:
            self.divsym = np.mean([self.div, np.fliplr(self.div)],axis=0)
            self.divasym = np.mean([self.div, -np.fliplr(self.div)],axis=0)

    def plot(self, src = 'div', ref = 2, init=4):

        _div = self.__dict__[src]
        v = self.source.v

        nspec = np.arange(len(self.div[init:]))
        plum_cycle = pplt.Cycle("viridis",len(nspec),lw=1.5)

        f2,a2 = pplt.subplots(refwidth=2.6, ncols=2, nrows=2, sharey=False, refaspect=1.4)
        for iv in _div[init:]:
            a2[0].semilogy(v,iv,cycle=plum_cycle)
            a2[0].set_xlabel("bias (mv)")
            a2[0].set_ylabel("didv",labelpad=-5)

        evec = [(np.log(aa)-np.log(_div[ref])+1) for aa in _div[init:]]

        evec_norm = [_e/np.mean(_e[0:10]) for _e in evec[3:]]

        for ee in evec:
            a2[1].plot(v, ee/np.mean(ee[-20:]),cycle=plum_cycle)
            a2[1].set_xlabel("bias (mv)")
            a2[1].set_ylabel(r"$\kappa / \kappa_{N}$")

        a2[1].set_ylim([.9*np.min(np.array(evec_norm)),1.1*np.max(np.array(evec_norm))])


        _tv = a2[2].imshow(np.array(evec_norm),cmap="balance",extent=[v[0],v[-1],len(_div),1],aspect=np.max(v)/(len(_div)-1))
        _cbar = a2[2].colorbar(_tv, title=r"$\kappa / \kappa_{N}$")

        for ee in evec:
            a2[3].plot(v, ee,cycle=plum_cycle)
            a2[3].set_xlabel("bias (mv)")
            a2[3].set_ylabel(r"$\kappa$")

        return f2

    def plot_differential(self, src = 'div', init=6, delta=5):

        _div = self.__dict__[src]
        v = self.source.v

        nspec = np.arange(len(self.div[init:]))
        plum_cycle = pplt.Cycle("viridis",len(nspec),lw=1.5)

        f2,a2 = pplt.subplots(refwidth=2.6, ncols=2, nrows=2, sharey=False, refaspect=1.4)
        for iv in _div[init:]:
            a2[0].semilogy(v,iv,cycle=plum_cycle)
            a2[0].set_xlabel("bias (mv)")
            a2[0].set_ylabel("didv",labelpad=-5)

        evec = [(np.log(_div[_id])-np.log(_div[_id-delta])+1) for _id in np.arange(init,len(_div),1)]

        evec_norm = [_e/np.mean(_e[0:10]) for _e in evec[3:]]

        for ee in evec:
            a2[1].plot(v, ee/np.mean(ee[0:10])+1,cycle=plum_cycle)
            a2[1].set_xlabel("bias (mv)")
            a2[1].set_ylabel(r"$\kappa / \kappa_{N}$")
        a2[1].set_ylim([.9*np.min(np.array(evec_norm)),1.1*np.max(np.array(evec_norm))])




        _tv = a2[2].imshow(np.array(evec_norm),cmap="balance",extent=[v[0],v[-1],len(_div),1],aspect=np.max(v)/(len(_div)-1))
        _cbar = a2[2].colorbar(_tv, title=r"$\kappa / \kappa_{N}$")

        for ee in evec:
            a2[3].plot(v, ee,cycle=plum_cycle)
            a2[3].set_xlabel("bias (mv)")
            a2[3].set_ylabel(r"$\kappa$")

        return f2
    
    
def slope_peek(spectra):
    try:
        ds1 = AndreevSlopes(datdict=spectra).FitGaussProcess().FitAndreevSlopes()
    except:
        print("fitting error")
        return None

    try:
        lw = 1.2
        subs = int(np.floor(ds1.source.dims["z"] / 10))

        f, a = pplt.subplots(nrows=2, ncols=3, share=False)

        with ds1.source as ds:
            a[0].plot(ds.v, ds.log_di[::subs].values.T)
            # a[1].plot(ds.v, ds.fit_di[::10].values.T)
            a[0].format(ylabel="log(di/dv)", xlabel="v, mV")
            a[1].plot(ds.v, ds.log_di[::subs].values.T, linewidth=lw, color="blue")
            a[1].plot(ds.v, ds.fit_di[::subs].values.T, alpha=0.5, linewidth=lw, color="red")

            for j in np.linspace(0, ds.dims["z"], 4, dtype=int):
                ds.log_di.isel(v=j).plot(ax=a[3], linewidth=0.5, color="blue")
                ds.fit_dlogi_dz.isel(v=j).plot(ax=a[3], linewidth=0.5, color="red")

            a[3].format(ylabel="log(di/dv)", xlabel=r"$\Delta$" + "z, nm", title="")

            for j in range(len(ds.z))[::subs]:
                (ds["slopes"].isel(z=j) / ds["slopes"].isel(z=j).sel(v=-3, method="nearest")).plot(ax=a[4],
                                                                                                   linewidth=0.5)
            a[4].format(xlabel="v, mV", ylabel=r"$\kappa/\kappa_{N}$", ylim=[0.5, 2.5], title="norm. slopes")

            normarr = []
            for r, j in zip(np.ones(ds.slopes.shape), ds.slopes.isel(v=3)):
                normarr.append(r * j.values)
            normarr = np.array(normarr)

            slopes = a[2].imshow(ds.slopes / normarr)
            a[2].colorbar(slopes, loc='r', shrink=ds.dims['z'] / ds.dims['v'])
            a[2].format(title="norm. slopes heatmap")

            errs = a[5].imshow(np.log10(ds.fit_di_mse))
            a[5].colorbar(errs, loc='r', shrink=ds.dims['z'] / ds.dims['v'])
            a[5].format(title="error heatmap")

            return f

    except:
        print("plotting error")
        return None
