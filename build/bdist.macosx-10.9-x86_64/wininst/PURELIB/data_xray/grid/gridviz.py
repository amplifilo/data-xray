from ..modules import *
from ..viz.core import Kaleidoscope, PlotRanger


def GridViewSpectra(ds=None, chan=['zf', 'zr'], ax=None, labels={'x':'', 'y':'', 't':''}, scale=['linear', 'linear'],
               pickrandom=False, plotindex=False, alpha=1, color=None, mod=lambda x: x, lim=[None, None],
               name=None, sf=None):
    """
    open spectra in the grid
    :param ds:
    :param chan:
    :param ax:
    :param labels:
    :param scale:
    :param pickrandom:
    :param plotindex:
    :param alpha:
    :param color:
    :param mod:
    :param lim:
    :param name:
    :param sf:
    :return:
    """
    #
    # options for scale:

    if pickrandom:
        if chan[0] in ds.keys():
            xsel = np.random.randint(ds[chan[0]].shape[0], size=pickrandom)
            ysel = np.random.randint(ds[chan[0]].shape[1], size=pickrandom)
            _spec = len(xsel)
        else:
            print('specify correct channel')
            return

    elif plotindex != False:
        # example would be np.where(xr.ds.cr_km==2)
        xsel, ysel = plotindex
        _spec = len(xsel)

    else:
        xsel, ysel = [np.ravel(i) for i in np.meshgrid(range(ds.dims['x']), range(ds.dims['y']))]
        _spec = ds.dims['x'] * ds.dims['y']

    framedim = (_spec, ds[chan[0]].shape[-1])

    for c, m in zip(Kaleidoscope(), chan):
        if m in ds.keys():
            ds_dat = ds[m].isel_points(x=xsel, y=ysel)
            dat = ds_dat.values.reshape(framedim).T
            # dat = np.reshape(p.values, framedim).T
            c = c if color is None else color
            ax.plot(ds.bias, mod(dat), c, linewidth=0.8, alpha=alpha, label=nonecheck(name))

    ax.set_xlabel(labelcheck(labels,'x'))
    ax.set_ylabel(labelcheck(labels,'y'))
    ax.set_title(labelcheck(labels,'t'))

    ax.set_xscale(scale[0])  # options: linear, log, symlog, logit
    ax.set_yscale(scale[1])  # options: linear, log, symlog, logit

    if lim[0] is not None:
        ax.set_xlim(lim[0])
    if lim[1] is not None:
        ax.set_ylim(lim[1])

    if sf is not None:
            plt.gcf().savefig(sf + '.png', dpi=300, bbox_inches='tight')

    return dat



def GridPlotSlices(ds=None, ind=[0], chan='cf', cfig=None, mod=lambda x: x, v=[0,3],
               ticks={'x': None, 'y': None}, labels={'x': '', 'y': '', 'ttl': ''}, sf=None):
    """
    Plot grid slices
    :param ds:
    :param ind:
    :param chan:
    :param cfig:
    :param mod:
    :param v:
    :param ticks:
    :param labels:
    :param sf:
    :return:
    """

        # side = int(np.sqrt(data.shape[0]))
        # if ind is None:
        #     slind = [int(i) for i in np.linspace(0, data.shape[1] - 1, 9)]
        # else:
        #     slind = ind

    _fig3 = plt.figure(figsize=(10,10)) if cfig is None else cfig
    dat = ds[chan]
    # fig3.suptitle('loading maps', size=11)
    G = gridspec.GridSpec(int(np.ceil(len(ind) / 3)), 3)
    for i, j in enumerate(ind):
        ax = _fig3.add_subplot(G[i])
        xname = list(ds.coords)[0]
        xval = int(ds[xname][j].values*10)/10
        ax.imshow(dat[:,:,j], cmap=plt.cm.RdBu)
        ax.set_title(xname + '  ' + str(xval), size=10)
    plt.show()

def GridViewHeatmapSNS(ds=None, chan='cr', mod=lambda x: x, v=1, ax=None, axis_labels=[None,None]):
    """
    Plot grid heatmap. It's nice, but needs fixing of the lables. No time

    :param ds:
    :param chan:
    :param mod:
    :param v:
    :param ax:
    :return:
    """
    import matplotlib
    
    def forceAspect(ax, aspect=1):
        im = ax.get_images()
        extent = im[0].get_extent()
        ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)

   

    if chan in ds.keys():

        # ax1.imshow(mod(ysrc), cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)

        bd = ds.dims['bias']
        xd = ds.dims['x']
        yd = ds.dims['y']

        
        data = mod(ds[chan].values.reshape((xd*yd, bd)))
        if type(v) is not list:
            v = PlotRanger(data, nsigma=v)

        data = pd.DataFrame(data, columns=[np.round(np.float(j), 1) for j in list(ds.bias.values)])
        #xticks = [np.float(ds.bias[int(x)]) for x in np.linspace(0,bd-1,10)]
        sns.heatmap(data=data, ax=ax, vmin=v[0],vmax=v[1], square=False)
        
        #ax1.set_yticks(yticks)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        


        plt.tight_layout()

def GridViewHeatmap(ds=None, chan='cr', mod=lambda x: x, v=[0, 3], \
                ticks={'x':None,'y':None}, labels={'x':'', 'y':'', 'ttl':''},
                sf=None, cfig=None):
    from matplotlib.ticker import NullFormatter
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    nmft = NullFormatter()

    def forceAspect(ax, aspect=1):
        im = ax.get_images()
        extent = im[0].get_extent()
        ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)

    if chan in ds.keys():

        # ax1.imshow(mod(ysrc), cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)

        bd = ds.dims['bias']
        xd = ds.dims['x']
        yd = ds.dims['y']

        _xt = ds.bias.values
        _yt = np.arange(xd * yd)

        _xt = [_xt[0], _xt[2]]
        _yt = [_yt[0], _yt[2]]

        if ticks['x'] is not None:
            _xt = [ticks['x'][0], ticks['x'][-1]]
        if ticks['y'] is not None:
            _yt = [ticks['y'][0], ticks['y'][-1]]

        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width + 0.02

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]

        # start with a rectangular Figure
        f2 = plt.figure(1, figsize=(8, 6)) if cfig is None else cfig

        rect_map = [left, bottom, width, height]
        rect_x = [bottom, bottom - 0.22, width * 0.8, 0.2]

        axX = plt.axes(rect_x)
        axMap = plt.axes(rect_map)

        axMap.xaxis.set_major_formatter(nmft)

        ysrc = ds[chan].values.reshape((xd * yd, bd))
        # _map = a2.pcolorfast(_xt, _yt, mod(ysrc[:,:-1]), cmap=plt.cm.RdBu)

        _map = axMap.pcolorfast(mod(ysrc), cmap=plt.cm.RdBu, vmin=min(v), vmax=max(v))

        # divider = make_axes_locatable(axMap)
        # cax = divider.append_axes('right', size='100%', pad=0.05)

        f2.colorbar(_map, orientation='vertical')

        axX.plot(ds.bias, 'b-')
        axX.set_ylabel(labels['x'], size=11)
        axMap.set_ylabel(labels['y'], size=11)

        axMap.set_title(chan if labels['ttl'] == '' else labels['ttl'])
        forceAspect(axMap, aspect=1)
        plt.autoscale(enable=True, axis=axX, tight=True)
        plt.axis('tight')

        if sf is not None:
            
            f2.savefig(sf + '.png', dpi=300, bbox_inches='tight')
       
        
        # if sf is not None:
        #     f2.savefig(sf + '.png', savedpi=200)

        #plt.tight_layout()
        #plt.show()