import numpy as np
import proplot as pplt
from nexusformat.nexus import nxload
from sklearn.decomposition import NMF
from scipy.signal import savgol_filter
from scipy.stats import hmean
from data_xray.file_io import GetData
from data_xray.report import QuickPPT
import xarray as xr
from scipy.signal import savgol_filter
from copy import copy
import os
import pandas as pd
import h5py


def despike(data, diff_thresh=30, hmean_window=15, savgol_parms={'window_length': 3, 'polyorder': 1}):
    xx = np.arange(len(data))
    diff = data - savgol_filter(x=data, window_length=savgol_parms['window_length'],
                                polyorder=savgol_parms['polyorder'])
    count = 0
    while len(xx[diff > diff_thresh]) and count < 10:
        for jx in xx[diff > diff_thresh]:
            data[jx] = hmean(data[jx - hmean_window:jx + hmean_window])
        diff = data - savgol_filter(x=data, window_length=savgol_parms['window_length'],
                                    polyorder=savgol_parms['polyorder'])
        count += 1
    return data


def pack_xr(clh5, remove_spikes=True):
    folder = os.path.dirname(clh5)
    base = os.path.basename(clh5)
    cl1_im2 = nxload(clh5)

    #print(cl1_im2.keys())
    if "Acquisition2" in cl1_im2.keys():
        x = np.array(cl1_im2['Acquisition2']['ImageData']['DimensionScaleC'])
        xen = 1.22779e-6 / x

        cl_data = cl1_im2['Acquisition2']['ImageData']['Image']
        cldata1 = np.array(cl_data).squeeze().swapaxes(0, -1)
        if remove_spikes:
            cldata1 = np.apply_along_axis(despike, -1, cldata1, diff_thresh=30, hmean_window=15,
                                       savgol_parms={'window_length': 3, 'polyorder': 1});
        print('here')
        
    clxr = xr.Dataset({"cl":
                           xr.DataArray(data=cldata1, dims=["x", "y", "en"], coords={
                               "x": np.arange(cldata1.shape[0]),
                               "y": np.arange(cldata1.shape[1]),
                               "en": xen}),
                       })

    return clxr


def nmf_cl(xrsrc, n_components=3, **kwargs):
    model = NMF(n_components=n_components, init='random', random_state=0, max_iter=1000)
    W1 = model.fit_transform(np.abs(xrsrc.cl.data.reshape(-1, 1024)))
    H1 = model.components_


    f2, a2 = pplt.subplots(nrows=len(H1), ncols=2, share=False, refwidth=2.5, refaspect=2.0)
    for j in range(W1.shape[-1]):
        a2[j, 0].imshow(W1[:, j].reshape(xrsrc.cl.data.shape[0:-1]), robust=True)
        a2[j, 0].format(title="NMF component " + str(j + 1))
        a2[j, 1].plot(xrsrc.en, H1[j])
        a2[j, 1].format(xlabel="photon energy (eV)", ylabel="rel. counts")

    # f2.savefig(folder+'/png/'+base+"nmf_.png", dpi=300, transparent=True)
    return [[W1, H1], f2]


def quick_cl_report(clh5):
    folder = os.path.dirname(clh5)

    if not (os.path.exists(folder + '/png')):
        os.mkdir(folder + '/png')

    base = os.path.basename(clh5)

    cl1_im2 = nxload(clh5)

    if "Acquisition2" in cl1_im2.keys():
        x = np.array(cl1_im2['Acquisition2']['ImageData']['DimensionScaleC'])
        xen = 1.22779e-6 / x

        cl_data = cl1_im2['Acquisition2']['ImageData']['Image']
        cldata1 = np.array(cl_data).squeeze().swapaxes(0, -1)
        cldata1 = np.apply_along_axis(despike, -1, cldata1, diff_thresh=30, hmean_window=15,
                                      savgol_parms={'window_length': 3, 'polyorder': 1});

        longvec = cldata1.shape[-1]
        #
        # x = np.arange(longvec)
        # cldata1 = np.apply_along_axis(savgol_filter,-1,cldata1,window_length=7,polyorder=2);

        _mean = cldata1.reshape(-1, longvec).mean(axis=0)
        _std = cldata1.reshape(-1, longvec).std(axis=0)

        # save histogram
        f3, a3 = pplt.subplots()
        a3.fill_between(xen, _mean - _std, _mean + _std, alpha=0.2, color="blue")
        a3.semilogy(xen, _mean, color="blue")

        a3.format(xlabel="nm", ylabel="counts")
        # f3.savefig(folder+'/png/'+base+"_2dhist_.png", dpi=300, transparent=True)

        return f2

        # NMF

        # model = NMF(n_components=3, init='random', random_state=0, max_iter=1000)
        # W1= model.fit_transform(np.abs(cldata1.reshape(-1,1024)))
        # H1 = model.components_

        # f2,a2 = pplt.subplots(nrows=len(H1), ncols=2, share=False, refwidth=2.5, refaspect=2.0)
        # for j in range(W1.shape[-1]):
        #     a2[j,0].imshow(W1[:,j].reshape(cldata1.shape[0:-1]), robust=True)
        #     a2[j,0].format(title="NMF component " + str(j+1))
        #     a2[j,1].plot(H1[j])
        #     a2[j,1].format(xlabel="nm", ylabel="counts")

        # #f2.savefig(folder+'/png/'+base+"nmf_.png", dpi=300, transparent=True)
        # return [f3, f2]

    else:
        return [None, None]