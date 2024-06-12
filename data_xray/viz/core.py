# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 10:37:38 2016

@author: peter

"""
from ..modules import *
from ..general_utilities import *
# from nanonis_io import *
# from nanonis_plotutils import *
import matplotlib.gridspec as gridspec
import holoviews as hv

def PlotRanger(df, nsigma = 1):
    dat = df/df.mean()
    return [np.float(v) for v in [dat.mean()- nsigma*dat.std(),dat.mean()+nsigma*dat.std()]]

def XarrayToHviews(darr):
    """
    #converts xarray DataArray object to HoloViews Dataset

    :param darr: Xarray DataArray (e.g. gr2.cf)
    :return:
    """
    d2 = [darr[j].values for j in darr.dims[::-1]]
    d2.append(darr.values)
    return hv.Dataset(tuple(d2), list(darr.dims[::-1]), darr.name)

def PrettyProplot(fig_font_scale=1.0, **kwargs):
    import proplot as pplt

    pplt.rc["axes.labelsize"]=fig_font_scale *14
    pplt.rc["axes.titlesize"]= fig_font_scale *16
    pplt.rc['fontname'] = 'Arial'
    pplt.rc['xtick.labelsize'] = fig_font_scale * 12
    pplt.rc['ytick.labelsize'] = fig_font_scale * 12
    pplt.rc["labelpad"]=0.7
    for key, value in kwargs.items():
        pplt.rc[key] = value
    # pplt.rc['fontname'] = 'Arial'
    # pplt.rc['font.size'] = fig_font_scale * 20
    # pplt.rc['axes.labelsize'] = fig_font_scale * 14
    # pplt.rc['axes.titleweight'] = 'bold'
    # pplt.rc['axes.titlesize'] = fig_font_scale * 14
    # pplt.rc['xtick.labelsize'] = fig_font_scale * 12
    # pplt.rc['ytick.labelsize'] = fig_font_scale * 12
    # pplt.rc['legend.fontsize'] = fig_font_scale * 10
    # pplt.rc['figure.titlesize'] = fig_font_scale * 14

def PrettyMatplotlib(fig_font_scale=1.0, **kwargs):
    import matplotlib as mpl

    mpl.rcParams["axes.labelsize"]=fig_font_scale *14
    mpl.rcParams["axes.titlesize"]= fig_font_scale *16
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['xtick.labelsize'] = fig_font_scale * 12
    mpl.rcParams['ytick.labelsize'] = fig_font_scale * 12
    mpl.rcParams["axes.labelpad"]=0.7
    for key, value in kwargs.items():
        mpl.rcParams[key] = value
    # pplt.rc['fontname'] = 'Arial'
    # pplt.rc['font.size'] = fig_font_scale * 20
    # pplt.rc['axes.labelsize'] = fig_font_scale * 14
    # pplt.rc['axes.titleweight'] = 'bold'
    # pplt.rc['axes.titlesize'] = fig_font_scale * 14
    # pplt.rc['xtick.labelsize'] = fig_font_scale * 12
    # pplt.rc['ytick.labelsize'] = fig_font_scale * 12
    # pplt.rc['legend.fontsize'] = fig_font_scale * 10
    # pplt.rc['figure.titlesize'] = fig_font_scale * 14

def SaveFigureAsImage(fileName, fig=None, **kwargs):
    ''' Save a Matplotlib figure as an image without borders or frames.
       Args:
            fileName (str): String that ends in .png etc.

            fig (Matplotlib figure instance): figure you want to save as the image
        Keyword Args:
            orig_size (tuple): width, height of the original image used to maintain
            aspect ratio.
    '''
    fig_size = fig.get_size_inches()
    w, h = fig_size[0], fig_size[1]
    fig.patch.set_alpha(0)
    upsample = kwargs['upsample'] if 'upsample' in kwargs.keys() else 1
    if 'orig_size' in kwargs.keys():  # Aspect ratio scaling if required
        w, h = kwargs['orig_size']
        w2, h2 = fig_size[0], fig_size[1]
        fig.set_size_inches([(w2 / w) * w, (w2 / w) * h])
        fig.set_dpi(upsample * (w2 / w) * fig.get_dpi())
        print(upsample)
    a = fig.gca()
    a.set_frame_on(False)
    a.set_xticks([]);
    a.set_yticks([])
    plt.axis('off')
    plt.xlim(0, h);
    plt.ylim(w, 0)
    fig.savefig(fileName, transparent=True, bbox_inches='tight', \
                pad_inches=0)
    print('...done')

def TextCoords(ax=None,scalex=0.9,scaley=0.9):
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    return {'x':scalex*np.diff(xlims)+xlims[0],
            'y':scaley*np.diff(ylims)+ylims[0]}

def CleanAxes(ax, lb='', xlim='', ylim='', fdict=None):

    if len(lb) > 0:
        ax.set_xlabel(lb[0], fontdict=fdict)
    if len(lb) > 1:
        ax.set_ylabel(lb[1], fontdict=fdict)
    if len(lb) > 2:
        ax.set_title(lb[2], fontdict=fdict)
    if len(xlim) > 0:
        ax.set_xlim(xlim)
    if len(ylim) > 1:
        ax.set_ylim(ylim)

#color lims nice
def ColorLimit(arr2d):
    return [np.mean(arr2d) - 3*np.std(arr2d), np.mean(arr2d) + 3*np.std(arr2d)]

def Kaleidoscope():
    from itertools import cycle
    return cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

# assumes a 2D array
def PlotArrayRandom(data, xvec=None, fx=lambda x: x, fy=lambda x: x, spread=lambda x, i=1: x * i / i, size=5,
                     labels=['x', 'y'], smooth=5):
    import scipy.signal as scp

    randind = np.random.randint(data.shape[0], size=size)
    _fig1, _ax = plt.subplots(1, 1)

    if xvec is None:
        xvec = np.arange(data.shape[1])

    for jk in randind:
        _ax.plot(fx(xvec), scp.savgol_filter(fy(spread(data[jk], jk)), window_length=smooth, polyorder=2))

        # _ax.plot(fx(xvec), wavelet_filter(data[jk],l=2, wlet='db6'))

        # if len(y_b) > jy:
        #    ax.semilogy(self.data['x'], np.abs(self.data[y_b[jk]][ix][iy]),'r-')
        _ax.set_xlabel(labels[0])
        _ax.set_ylabel(labels[1])

    plt.tight_layout()
    plt.show()


def PlotArraySlice(data, ind=None):
    side = int(np.sqrt(data.shape[0]))
    if ind is None:
        slind = [int(i) for i in np.linspace(0, data.shape[1] - 1, 9)]
    else:
        slind = ind

    _fig3 = plt.figure()
    # fig3.suptitle('loading maps', size=11)
    G = gridspec.GridSpec(int(np.ceil(len(slind) / 3)), 3)
    for i, j in enumerate(slind):
        ax = _fig3.add_subplot(G[i])
        ax.imshow(data[:, j].reshape(side, side), cmap=plt.cm.RdBu)
        ax.set_title('slice #' + str(j), size=10)
    plt.imshow()

def GetFigureData(artists=None, axes=None, **kwargs):
    from matplotlib import _pylab_helpers as pylab_helpers
    from matplotlib import cbook
    from mpldatacursor import HighlightingDataCursor
    xdata = list()
    ydata = list()

    def plotted_artists(ax):
        artists = (ax.lines + ax.patches + ax.collections
                   + ax.images + ax.containers)
        return artists

    # If no axes are specified, get all axes.
    if axes is None:
        managers = pylab_helpers.Gcf.get_all_fig_managers()
        figs = [manager.canvas.figure for manager in managers]
        axes = [ax for fig in figs for ax in fig.axes]

    if not cbook.iterable(axes):
        axes = [axes]

    # If no artists are specified, get all manually plotted artists in all of
    # the specified axes.
    if artists is None:
        artists = [artist for ax in axes for artist in plotted_artists(ax)]
    for a in artists:
        xdata.append(a.get_xdata())
        ydata.append(a.get_ydata())
    return xdata, ydata

def SaveFigure(figobj, figpath=None, figname='t1'):
    if figpath is None:
        figpath = os.getcwd()
    figobj.savefig(figpath + '/' + figname + '.svg', format='svg')
    figobj.savefig(figpath + '/' + figname + '.png', format='png')
    pyperclip.copy(figpath + '/' + figname + '.png')

def hldatacursor(artists=None, axes=None, **kwargs):
    from matplotlib import _pylab_helpers as pylab_helpers
    from matplotlib import cbook
    from mpldatacursor import HighlightingDataCursor

    def plotted_artists(ax):
        artists = (ax.lines + ax.patches + ax.collections
                   + ax.images + ax.containers)
        return artists

    # If no axes are specified, get all axes.
    if axes is None:
        managers = pylab_helpers.Gcf.get_all_fig_managers()
        figs = [manager.canvas.figure for manager in managers]
        axes = [ax for fig in figs for ax in fig.axes]

    if not cbook.iterable(axes):
        axes = [axes]

    # If no artists are specified, get all manually plotted artists in all of
    # the specified axes.
    if artists is None:
        artists = [artist for ax in axes for artist in plotted_artists(ax)]

    return HighlightingDataCursor(artists, **kwargs)

