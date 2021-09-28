#%%
'''import functions'''

import data_xray.file_io as fio
from data_xray.modules import *
from data_xray.report import SummaryPPT
from data_xray.nanonisio import Spectrum
import xarray as xr
import pandas as pd
import os

# %%
sourceFolder = "~/"
# %%
sourceFolder = fio.GetData.select_folder()

#%%
sxmdict = fio.GetData.find_data(topdir=sourceFolder, ext='sxm', get_data=True)
#%%
pres = SummaryPPT(fdict=sxmdict, pname="imageSummary_Z", chanselect = {'scan':'Z'})
# %%
'''
what about .dat files (single spectra?)
'''

#%%
sourceFolder = fio.GetData.select_folder()

# %%
sourceFolder
# %%
datdict = fio.GetData.find_data(topdir=sourceFolder, ext='dat', get_data=False)

# %%
datdict
# %%
datdict[0]
# %%
s1 = Spectrum(datdict[0])

#%%
def packIV(fname):
    s1 = Spectrum(fname);
    z = [np.float(s1.header["Z (m)"]) , np.float(s1.header["Z offset (m)"])];
    cur = np.mean([s1.signals["Current (A)"], s1.signals["Current [bwd] (A)"]], axis=0);
    bias = s1.signals["Bias calc (V)"];
    return [z, bias, cur]

#%%
spectra = [packIV(z) for z in datdict];

#%%
set([1,0,1])
#%%
np.polyfit

#%%
set([len(s[1]) for s in spectra])

#%%
zarr = [np.sum(s[0]) for s in spectra]
zarr = (zarr - np.min(zarr))/1e-10
varr = np.mean([s[1] for s in spectra], axis=0)/1e-3
iarr = np.asarray([s[-1] for s in spectra])

#%%
iArray = xr.DataArray(data=iarr, dims=["z","v"], coords={"z":zarr, "v":varr})/1e-9
#%%
np.exp(40)
#%%
plt.imshow(np.log(abs(iArray)))

#%%
iArray[:,0].plot()

#%%

np.log(iArray[:,0]).plot()
#%%
plt.plot(np.diff(np.log(iArray[:,0])))

#%%
f2, a2 = plt.subplots()
for j in range(0,iArray.shape[0],4):
    np.log(np.abs(iArray[:,j])).plot(ax=a2)   

#%%
test_ind =60
z_test =  np.linspace(np.min(iArray.z), np.max(iArray.z),100)
f = np.polyfit(x=iArray[:,test_ind].z, y=np.log(iArray[:,test_ind].data),deg=4)
f2, a2 = plt.subplots()
np.log(iArray[:,test_ind]).plot(ax=a2)
a2.plot(z_test,np.polyval(f,z_test))
f
# %%
z_test =  np.linspace(np.min(iArray.z), np.max(iArray.z),1000)
ifit = []
d_ifit = []

for j in range(iArray.shape[1]):
    f = np.polyfit(x=iArray[:,j].z, y=np.log(np.abs(iArray[:,j].data)),deg=6)
    fval = np.polyval(f,z_test)
    ifit.append(fval)
    d_ifit.append(np.gradient(fval)/np.gradient(z_test))
#%%
iFitArray = xr.DataArray(data=np.asarray(ifit), dims=["v","z"], coords={"v":varr,"z":z_test})
d_iFitArray = xr.DataArray(data=np.asarray(d_ifit), dims=["v","z"], coords={"v":varr,"z":z_test})

#%%
iFitArray.sel(v=0.07,method="nearest").plot()
#%%

z_test =  np.linspace(np.min(iArray.z), np.max(iArray.z),1000)

plt.plot(np.gradient(np.polyval(f,z_test)))
# %%
varr/1e-3
# %%
'''need to now make xarray of numerical derivatives ''' 
#%%
f2, a2 = plt.subplots()
np.log(iArray.sel(v=0.00, method="nearest")).plot(ax=a2)
iFitArray.sel(v=0.00, method="nearest").plot(ax=a2)

#%%
f2, a2 = plt.subplots()
d_iFitArray.sel(v=+0.1, method="nearest").plot(ax=a2)
#%%
def norm(yy):
    return yy/yy[0]

#%%
norm(d_iFitArray.sel(z=0.05, method="nearest")).plot()
plt.ylim(-.5, 2.5)
# %%
xx = np.arange(0,5,0.1)
yy = xx*2+10
plt.plot(xx,yy)

np.gradient(yy)/np.gradient(xx)
# %%
