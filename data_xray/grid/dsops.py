from ..cluster import *
from ..modules import *

#############################################################
##############################################################
# Clustering functions
"""
this is a variety of scripts that channel sckikit
"""

"""GridViz is the grid visualization/manipulation too.
initial data is imported as a netcdf sourced xarray, or passed
diretly as an xarray
"""

#just a comment
#xr below is just a GridXarray object
#functions specific to the manipulation of the xarray dataset\

def GetBiasIndexDS(ds,bias=[0]):
    """
    get index of channels within a dataset corresponding to specific bias

    :param ds: xarray dataset
    :param bias: sought values for the main coordinate of the dataset
    :return: indices of the values
    """

    inds = []
    for b in bias:
        inds.append(nearest(ds.bias.values, b))
    return inds

def FindIndexDS(ds,chan,val):
    """
    simple function to get the indices of specific values. Primarily for use with clustering. e.g.
    which pixels correspond to specific cluster number

    :param ds: xarray dataset
    :param chan: channel
    :param val: sought values
    :return:
    """

    return np.where(ds[chan].values.reshape((ds.dims['x'] * ds.dims['y'], 1)) == val)[0]

def ChanHistogramDS(ds, xy=['bias', 'cf'], xymod=[lambda x: x, lambda x: x], ax=0, label=['', '', ''], px=None):
    
    from matplotlib.colors import LogNorm 
    """
    plot histogram of a specific channel within the xarray dataset
    px 1D list of indices to sub-select from the dataset. for example ds1.zf_kmeans.values.reshape((ds1.dims['x']*ds1.dims['y'],1))==2

    :param ds: histogram of a specific channel in the xarray
    :param xy: list containing x and y coordinates
    :param xymod: if desired, pass the modifier for the plotting of the histogram via lambda functions
    :param ax: axis to plot in
    :param label: labels of the histogram
    :param px: select pixels
    :return:
    """

    x = xymod[0](ds[xy[0]].values)
    y = xymod[1](ds[xy[1]].values)
    x = np.ones(y.shape) * x


    if px is None:
        y = y.reshape((y.shape[0] * y.shape[1] * y.shape[2]))
        x = x.reshape((x.shape[0] * x.shape[1] * x.shape[2]))
    else:

        y = y.reshape((ds.dims['x'] * ds.dims['y'], ds.dims['bias']))
        x = x.reshape((ds.dims['x'] * ds.dims['y'], ds.dims['bias']))

        y = y[px]
        x = x[px]

        y = y.reshape((y.shape[0] * y.shape[1]))
        x = x.reshape((x.shape[0] * x.shape[1]))

    a2 = pd.DataFrame(y, columns=['y'])
    a2['x'] = x
    #         #
    if ax == 0:
        print('please provide a valid axis')
    else:
        a = ax.hist2d(a2.x, a2.y, bins=(100, 100), normed=1, norm=LogNorm(), cmap=plt.cm.jet)
        ax.set_xlabel(label[0])
        ax.set_ylabel(label[1])
        ax.set_title(label[2])
        plt.colorbar(a[3])

def ChanPcaDenoiseDS(ds, xvec='bias', chan='cf',keep_components=4):
    """
    denoise specific channel with simple PCA-cut off filter

    :param ds: xarray dataset
    :param xvec: name of dataset coordinate
    :param chan: selected channel
    :param keep_components: maximum principal component to keep. Weights of higher components will be nullified in reconstruction
    :return: injects a dedocated field into xarray with denoised channel
    """

    cf = ds[chan].values
    cf = cf.reshape((ds.x.shape[0] * ds.y.shape[0], ds[xvec].shape[0]))
    cf_dn = PcaDenoise(pca_dat=cf, keep_components=keep_components)
    cf_dn = cf_dn.reshape((ds.x.shape[0], ds.y.shape[0], ds[xvec].shape[0]))
    ds[chan+'_dn'] = (ds[chan].dims,cf_dn)
    ds.attrs.update([(chan+'_dn', 'pca denoise of ' + chan)])
    
    
def ChanPcaKmeansDS(ds, xvec='bias', chan='cf', mod=lambda y: y, comps=6,nclust=4, fig=None):
    """
   
    :param ds: xarray dataset
    :param xvec: name of dataset coordinate
    :param chan: selected channel
    :param mod: if desired, pass the modifier for the plotting of the histogram via lambda functions
    :param comps: number of pca components to keep for clustering (which will be performed on the scores)
    :param nclust: number of clusters 
    :param fig: figure to plot in
    :param xsel: 
    :return:
    """


    cf2 = ds[chan]
    cf = mod(cf2).values

    cf = cf.reshape((cf2.x.shape[0] * cf2.y.shape[0], cf2[xvec].shape[0]))
    km, _ = PcaKmeans(X=cf, dims=comps, nclust=nclust, fig=fig)
    km = km.labels_.reshape((cf2.x.shape[0], cf2.y.shape[0]))
    ds[chan + '_kmeans'] = (ds[chan].dims[:-1],km)
    ds.attrs.update([(chan + '_kmeans', 'kmeans clustering of ' + chan + ' with')])
    return km

# def ChanEarthSlopeDS(ds, chan=['zf', 'zr'], fitpx=None, thresh=0.97, incr=0.5):
#     """
#      fit specific channel with polyline fitting via multivariate adaptive regression splines
#     thresh=thresh,incr=incr
#     :param ds:
#     :param chan:
#     :param fitpx:
#     :param thresh:
#     :param incr:
#     :return:
#     """

#     m = Earth()
#     xvec = ds.bias.values
#     if fitpx is None:
#         fitpx = [ds.x, ds.y]
#     for c in chan:
#         ds.__setitem__(c + '_earth', ds[c])
#         ds[c + 'efit'] = ds[c] - ds[c]  # reset the fit container to null
#         for xi in tqdm(fitpx[0]):
#             for yi in tqdm(fitpx[1]):
#                 yvec = ds[c][xi, yi].values
#                 m.fit(xvec, yvec)
#                 ds[c + '_earth'][xi, yi] = m.predict(xvec)

def ChanModDS(ds, chan, mod=lambda x: x):
    """
    apply any well-behaved lambda function to a specific channel within a dataset
    :param ds: xarray dataset
    :param chan: channel to apply modification too
    :param mod: well-behaved lambda function
    :return: adds a channel with the modified data
    """


    for m in chan:
        if m in ds.keys():

            ds[m] = mod(ds[m])
            print('modifying ' + m + '..all done!')
            return mod(ds[m])

def SubtractZOffsetDS(ds=None, chan=['zr', 'zf']):
    """
    subtract offset from any channel defined by the first point on the curve
    :param ds: xarray dataset
    :param chan: channels to subtract offsets from
    :return:
    """

    try:
        ChanModDS(ds, chan=chan, mod=lambda x: (x - x[:, :, 0]))
        print('z_subtract..all done!')
    except:
        print('..no changes made. Check params')

def CurrentOffsetSubtractDS(ds=None, chan=['cf', 'cr'], vrange=None):
    """
    subtract offset from any channel within a specified range of bias (or other x-coordinate)
    :param ds: xarray dataset
    :param chan: channel to subtract offset from
    :param vrange: substract offset from specific range of the coordinate vector
    :return:
    """

    if vrange != None:
        for m in chan:
            if m in ds.keys():
                _offset = ds[m].isel(
                    bias=np.where((ds.bias < max(vrange)) * (ds.bias > min(vrange)))[0])
                _offset = _offset.mean(axis=-1)
                ds[m] = ds[m] - _offset
        print('iv_offset ..all done!')
    else:
        print('..no changes made. Check params')

def ChanDerivativeDS(ds=None,chan='cf',xvec='bias'):
    """
    take derivative of a specific channel
    :param ds: xarrat dataset
    :param chan: channel to take derivative of
    :param xvec: name of the dataset coordinate
    :return: appends derivative to the dataset
    """

    cf = ds[chan].values
    dcf = np.gradient(cf, np.min(np.diff(ds1[xvec].values)), axis=2)

    ds[chan + '_nder'] = (ds[chan].dims, dcf)
    ds.attrs.update([(chan + '_nder', 'numerical derivative of ' + chan)])


def ClusterCenter(xr=None, chan='cr'):
    """
    ##this function needs to expand to accomodate more clustering algorithms
    Obtain centers of clusters after applying ChanPcaKmeansDS or similar functions
    :param xr: xarray
    :param chan:
    :return:
    """
    cent_dict = {}
    if chan + '_km' in xr.ds:
        cls = np.arange(np.max(xr.ds[chan + '_km']) + 1) #looks for identifier of the clustered field
        for j in cls:
            f, a = plt.subplots(1, 1)
            xs, ys = np.where(xr.ds.cr_km == j)
            dat = xr.ds.cr.isel_points(x=xs, y=ys)
            xm = np.abs(xr.ds.bias)
            ym = dat.mean(axis=0)
            yerr = dat.std(axis=0)
            cent_dict[j] = [ym, yerr]
            a.errorbar(np.abs(xm), ym, yerr)
            a.set_xscale("log", nonposx='clip')
            a.set_yscale("log", nonposy='clip')
            # a.set_xlim([6e-1, np.max(xm)])
            a.set_xlabel('bias, V')
            a.set_ylabel('current, nA')
            a.set_title('cluster #' + str(j))
    plt.show()
    return cent_dict


