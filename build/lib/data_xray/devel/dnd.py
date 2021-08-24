def PlotScanLy(_Scan, chans='all', scanid=0):
    import plotly.tools as tls
#     matplotlib.rcParams['font.size'] = 8
#     matplotlib.rcParams['font.family'] = ['sans-serif']
#     # chans two values 'fil' or 'all'. All - just plot all channels
#     # ppl - means use pyplot interface
#     # 'fil' - plot ony those with meaningful std (aka data). Std threshdol 1e-2
#     # the plotter! plots all channels contained in the sxm-dict structure
#
#
    fn = _Scan.fname
    #plotdict = dict()
    # allow plotting of specific channels
    if chans == 'all':
        chans = _Scan.signals.keys()
    elif chans == 'fil':
        chans = []
        for c in _Scan.signals.keys():
            im2 = _Scan.signals[c]['forward']
            im2 = im2/(np.max(im2)-np.min(im2))
            if np.std(im2) > 0.2:
                chans.append(c)

    if len(chans) == 0:
        return #no valid data found

    fig3 = tools.make_subplots(rows=2, cols=len(chans), vertical_spacing=0.5)


    #fig3, ax3 = plt.subplots(nrows=2, ncols=len(chans),figsize=(5,6*(1+len(chans))))
        #, sharex='col', sharey=True,
        #                     gridspec_kw={'height_ratios': [2, 1]},
        #                     figsize=(4, 7))
    # if len(chans) == 1:
    #     ax3 = ax3[:,np.newaxis]

    for j, ch in tqdm(enumerate(chans)):
        traceForw = PlotImageLy(_Scan, chan=ch, ax=None, forw=1)
        traceRev = PlotImageLy(_Scan, chan=ch, ax=None, forw=0)

        fig3.append_trace(traceForw,1,j+1)
        fig3.append_trace(traceRev, 2,j+1)

        #ax3[0,0] = f3.add_subplot(gs[0, j])
        #axr = f3.add_subplot(gs[1, j])
    fig3['layout'].update(height=800, width=400*(len(chans)), title="Scan #"+ str(scanid) + ':' + fn)
    return fig3

    #plt.tight_layout(pad = 0.04, w_pad = 0.05, h_pad = 0.05)
    #plt.subplots_adjust(left=0.0, bottom=0.2, right=0.1, top=0.9, wspace=0.2, hspace=0.2)
    #plt.tight_layout(pad=0)
    #fig3.set_tight_layout({'rect': [0, 0.0, 1, 0.92], 'pad': 0.05, 'h_pad': 0.05})
    #plt.setp(axes, title='Test')
    #fig3.suptitle("Scan #"+ str(scanid) + ':' + fn, size=12)

def PlotImageLy(_Scan, chan, ax, forw=1, cm='magma'):

    #clean up missing vavlues
    from matplotlib.colors import LogNorm
    fb = 'forward' if forw else 'backward'
    im2 = _Scan.signals[chan][fb]
    if np.any(np.isnan(im2)):
        cleanrange = np.setxor1d(np.arange(np.size(im2, 0)), np.unique(np.where(np.isnan(im2))[0]))
        im2 = im2[cleanrange]

    if chan.lower()=='z':
        im2 = SubtractLine(im2)

    scale = _Scan.header['scan_range']/_Scan.header['scan_pixels']
    stat = [np.mean(im2), np.mean([np.std(im2 / np.max(im2)), np.std(im2 / np.min(im2))])]
    color_range = [stat[0] - 3 * stat[1], stat[0] + 3 * stat[1]]
    #im2 = (im2 - np.min(im2))/(np.max(im2)-np.min(im2))
    im2 = (im2 - np.min(im2))+1e-4# / (np.max(im2) - np.min(im2))
    im2 = im2/np.max(im2)
    #im2 = np.log10(im2)
    #trace = go.Heatmap(z=im2.tolist())
    trace = go.Heatmap(z=im2.tolist(), colorscale='YlOrRd')
    return trace
    #fig = go.Figure(data=[trace])
    # ax.imshow(im2, cmap=cm, norm=LogNorm(vmin=0.05, vmax=1.1))
    #
    # scale_unit = np.min([i / j for i, j in zip(_Scan.header['scan_range'], im2.shape)])
    #
    # sbar = ScaleBar(scale_unit, location='upper right', font_properties={'size': 8})
    #
    # fp = sbar.get_font_properties()
    # fp.set_size(8)
    # #ax.axes([0.08, 0.08, 0.94 - 0.08, 0.94 - 0.08])  # [left, bottom, width, height]
    # #ax.axis('scaled')
    # ax.add_artist(sbar)
    # ax.set_title(chan+' '+ fb, fontsize=8)
    # ax.set_title(phdf+j)
    # ax.set_title('$'+j+'$')
    #ax.set_axis_off()




def gridKmeans(gf, chan, nclust=5):
    from sklearn.cluster import KMeans
    dat = gf.data[chan]
    dat2 = np.reshape(dat, (np.size(dat, 0) * np.size(dat, 1), np.size(dat, 2)))
    km = KMeans(n_clusters=nclust)
    km.fit(dat2)
    clustind = km.predict(dat2)
    meanvec = list()
    for j in range(nclust):
        clustrep = list(np.where(clustind == j)[0])
        meanvec.append(np.mean(dat2[clustrep, :], 0))
    return clustind, meanvec


def _earth_iv(self, curchan='Current', plotall=0, plotfit=1):
    # self should be gridfile() object
    def earth_slope(xvec, yvec, plot=0):
        m = Earth()

        sigind = np.where(np.log10(np.abs(yvec)) > p['thresh'] * self.allnoise)
        vpos = np.intersect1d(sigind, np.where(xvec > 0))
        vneg = np.intersect1d(sigind, np.where(xvec < 0))

        posx, posc, negx, negc = np.abs(xvec[vpos]), np.abs(yvec[vpos]), np.abs(xvec[vneg]), np.abs(yvec[vneg])
        tl = np.zeros(xvec.shape)
        l1, l2 = [], []
        rsq = []

        if len(posc) > 4:  # a meaningful length of iv curve
            m.fit(np.log(posx), np.log(posc))
            l1 = m.predict(np.log(posx))
            tl[vpos] = np.gradient(l1) / np.gradient(np.log(posx))
            rsq.append(m.rsq_)
            if p['plotfit']:
                p['plotax'].plot(np.log(posx), incrj * p['incr'] + l1, linewidth=0.4, color='red')
                p['plotax'].scatter(np.log(posx), incrj * p['incr'] + np.log(posc), s=10, facecolors='none',
                                    edgecolors='blue')

        if len(negc) > 4:
            m.fit(np.log(negx), np.log(negc))
            l2 = m.predict(np.log(negx))
            tl[vneg] = np.gradient(l2) / np.gradient(np.log(negx))
            rsq.append(m.rsq_)
            if p['plotfit']:
                p['plotax'].plot(np.log(negx), incrj * p['incr'] + l2, linewidth=0.4, color='red')
                p['plotax'].scatter(np.log(negx), incrj * p['incr'] + np.log(negc), s=10, facecolors='none',
                                    edgecolors='blue')

        return tl, np.mean(rsq)

        tl = np.zeros(xvec.shape)
        m.fit(xvec, yvec)
        yh = m.predict(xvec)
        dyh = np.abs(np.gradient(yh) / np.gradient(xvec))
        datind = np.where(dyh > np.min(dyh) * 1.1)
        xvec = xvec[dyh]
        yvec = yvec[dyh]

        # np.where(~np.isnan(np.log(yvec)) == True)[0][10:]
        posx, posc, negx, negc = xvec[xvec > 0], yvec[xvec > 0], np.abs(xvec[xvec < 0]), yvec[xvec < 0]
        if plot:
            fig5 = plt.gcf()
            ax5 = fig5.add_subplot(111)

        if len(posx):
            m.fit(np.log(posx), np.log(posc))
            l1 = m.predict(np.log(posx))
            tl[np.where(xvec > 0)] = np.gradient(l1) / np.gradient(np.log(posx))
            if plot:
                ax5.plot(np.log(posx), l1, 'g--')
                ax5.plot(np.log(posx), np.log(posc), 'b.')

        if len(negx):
            m.fit(np.log(negx), np.log(negc))
            l2 = m.predict(np.log(negx))
            tl[np.where(xvec < 0)] = np.gradient(l2) / np.gradient(np.log(negx))
            if plot:
                ax5.plot(np.log(negx), l2, 'g--')
                ax5.plot(np.log(negx), np.log(negc), 'b.')

        return tl


def GroupBySimilarity(xr=None, xrdat=None, nclust=3, plotit=1, clusterorder=None):
    # let's revise this that data can be shaped externally

    def get_dimensions(xrobj):

        dim_names = []
        for k in list(xrobj.dims):
            if k not in list(xrobj.coords.keys()):
                dim_names.append(k)

        dim_sizes = [xrobj[i].shape[0] for i in dim_names]
        size_arr = [np.size(xrobj) / xrobj.bias.shape[0], xrobj.bias.shape[0]]
        return dim_names, dim_sizes, size_arr

    chan = xrdat.name
    clust_dim_names, clust_dim_sizes, clust_size_arr = get_dimensions(xrdat)
    src_dim_names, src_dim_sizes, src_size_arr = get_dimensions(xr.ds[chan])

    # clust_dim_sizes = [xrdat[i].shape[0] for i in dim_names]
    # clust_size_arr = [np.size(xrdat)/xrdat.bias.shape[0],xrdat.bias.shape[0]]
    #
    # src_dim_sizes = [xr.ds[chan].shape[0] for i in dim_names]
    # src_size_arr = [xr.ds[chan].shape[0] for i in dim_names]
    #

    yclust = xrdat.values.reshape(tuple(int(s) for s in clust_size_arr))
    ysrc = xr.ds[chan].values.reshape(tuple(int(s) for s in src_size_arr))

    km = pca_kmeans(ysrc, nclust=nclust, plotit=0)

    kmarr = xr.DataArray(np.reshape(km.labels_, tuple(clust_dim_sizes)), dims=clust_dim_names)

    """"""

    mean_dict = []
    err_dict = []

    for j in np.arange(nclust):
        dat = ysrc[np.where(km.labels_ == j)]

        # print(np.where(kmarr==j))
        # play with np.flatten
        # xs, ys = np.where(kmarr == j)
        # dat = xr.ds[chan].isel_points(x=xs, y=ys)
        mean_dict.append(np.abs(dat.mean(axis=0)))
        err_dict.append(dat.std(axis=0))

    if clusterorder is None:
        cluster_order = np.flipud(np.argsort(np.asarray(mean_dict).mean(axis=1)))
    else:
        cluster_order = clusterorder
    # print(cluster_order)
    # print(km.labels_)
    kmsort = copy.copy(km.labels_)
    for j in np.arange(np.max(km.labels_) + 1):
        kmsort[np.where(km.labels_ == j)] = cluster_order[j]

    # kmarr = xr.DataArray(np.reshape(kmsort, (x, y)), dims=['x', 'y'])
    kmarr = xr.DataArray(np.reshape(kmsort, tuple(clust_dim_sizes)), dims=clust_dim_names)

    xr.ds.__setitem__(chan + '_km', kmarr)

    mean_dict = np.asarray(mean_dict)[np.argsort(np.asarray(mean_dict).mean(axis=1))]
    err_dict = np.asarray(err_dict)[np.argsort(np.asarray(mean_dict).mean(axis=1))]

    cm = xr.DataArray(data=mean_dict.T, coords={'bias': xr.ds.bias.values, chan + '_clust': np.arange(nclust)},
                      dims=['bias', chan + '_clust'],
                      name=chan + '_clust_mean')
    cmerr = xr.DataArray(data=err_dict.T, coords={'bias': xr.ds.bias.values, chan + '_clust': np.arange(nclust)},
                         dims=['bias', chan + '_clust'], name=chan + '_clust_err')

    xr.ds.__setitem__(chan + '_km' + '_mean', cm)
    xr.ds.__setitem__(chan + '_km' + '_err', cmerr)

    if plotit:
        # scree plot
        for i in range(nclust):
            k_ind = np.where(km.labels_ == i)[0]
            np_glance_random(data=xr.ds[xrdat.name], xvec=xr.ds.bias.values, labels=['V', 'z', str(i)])



class Plot(object):
    # use case scenarious:
    # z = Plot([np.asarray([y,y**3,y**5]).T,y],1,lp=['','b.-'],gdim=[2,1])
    # z = Plot(y**5,1,lp='g.--')

    def __init__(self, y=None, show=None, **kwargs):

        if y is None:
            print('specify y-vector')
            return

        listORsingle = lambda rp: rp if isinstance(rp, list) else [rp]
        self.y = listORsingle(y)
        self.y = [listORsingle(j) for j in y]  # a sneakish way to encapsulate y-plots

        self.kwargs = kwargs
        self.show = show

        # here we need to set the subplot structure
        self._set_subplots()
        self._plotit()

        if self.show is not None:
            plt.show()
            # self._f = plt.gcf()

    def _from_dict(self, d):
        for k, v in d.items():
            setattr(self, k, d[k])

    def _to_dict(self):
        d = dict()
        for k, v in self.__dict__.items():
            d[k] = v
        return d

    def _sel(self, x, alt):
        return self.kwargs[x] if x in self.kwargs.keys() else alt

    def _set_default(self, prop, v=lambda *args: []):

        self.__dict__[prop] = self._sel(prop, [])
        if len(self.__dict__[prop]) == 0:
            for ya in self.y:
                # pdb.set_trace()
                self.__dict__[prop].append(v(ya))

    def _clone_labels(self, prop):
        listORsingle = lambda rp: rp if isinstance(rp, list) else [rp]

        if len(listORsingle(self.__dict__[prop])) != np.prod(self.gdim):
            val = listORsingle(self.__dict__[prop])[0]  # first value considered default
            self.__dict__[prop] = []
            for j in range(np.prod(self.gdim, dtype=int)):
                self.__dict__[prop].append(val)

    def _set_subplots(self, set_default=True):

        self.gdim = self._sel('gdim', None)
        if self.gdim is None:
            self.gdim = (1, len(self.y))  # just a row


        if 'ax' in self.kwargs.keys():
            self.ax = self.kwargs['ax']  # this is important - will work even if subplot structure is completely ignored
        else:
            self._f = plt.figure(figsize=self._sel('figsize', (4.2 * self.gdim[1], 4 * self.gdim[0])))
            gs = gridspec.GridSpec(*self.gdim)
            self.ax = [plt.subplot(i) for i in gs]  # line them up

        def_list = lambda val, a: [val for m in a]
        if set_default:
            default_dict = {'x': lambda *a: [np.arange(len(m)) for m in a[0]],
                            'lp': lambda *a: def_list('b.-', a[0]),
                            'xlabl': lambda *a: '',
                            'ylabl': lambda *a: '',
                            'ptitles': lambda *a: '',
                            'ymod': lambda *a: 'lambda y: y',
                            'plabl': lambda *a: def_list('', a[0]),
                            'yerr': lambda *a: def_list(None, a[0])}

            for k, d in default_dict.items():
                self._set_default(k, v=d)

        for s in ['ylabl', 'xlabl', 'ptitles']:  # in case of lazy labeling
            self._clone_labels(s)

    def _plotit(self):

        listORsingle = lambda rp, jj: rp[jj] if isinstance(rp, list) else rp

        for j, yj in enumerate(self.y):

            for jj, yjj in enumerate(yj):
                x = listORsingle(self.x[j], jj)  # self.x[j][jj]
                ym = listORsingle(self.ymod[j], jj)
                y = eval(ym)(yjj)
                lp = listORsingle(self.lp[j], jj)

                plabel = self.plabl[j][jj]
                ye = self.yerr[j][jj]
                self.ax[j].errorbar(x, y, yerr=ye, fmt=lp, label=plabel)

            self.ax[j].set_xlabel(self.xlabl[j], size=11)
            self.ax[j].set_ylabel(self.ylabl[j], size=11)
            self.ax[j].set_title(self.ptitles[j], size=11)

            plt.legend()

    def _replot(self):

        self._set_subplots(set_default=False)
        self._plotit()
        #plt.show()

class PointBrowser(object):
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points

     # if __name__ == '__main__':
    #     import matplotlib.pyplot as plt
    #
    #     X = np.random.rand(100, 200)
    #     xs = np.mean(X, axis=1)
    #     ys = np.std(X, axis=1)
    #
    #     fig, (ax, ax2) = plt.subplots(2, 1)
    #     ax.set_title('click on point to plot time series')
    #     line, = ax.plot(xs, ys, 'o', picker=5)  # 5 points tolerance
    #
    #     browser = PointBrowser()
    #
    #     fig.canvas.mpl_connect('pick_event', browser.onpick)
    #     fig.canvas.mpl_connect('key_press_event', browser.onpress)
    #
    #     plt.show()

    """

    def __init__(self):
        self.lastind = 0

        self.text = ax.text(0.05, 0.95, 'selected: none',
                            transform=ax.transAxes, va='top')
        self.selected, = ax.plot([xs[0]], [ys[0]], 'o', ms=12, alpha=0.4,
                                 color='yellow', visible=False)

    def onpress(self, event):
        if self.lastind is None:
            return
        if event.key not in ('n', 'p'):
            return
        if event.key == 'n':
            inc = 1
        else:
            inc = -1

        self.lastind += inc
        self.lastind = np.clip(self.lastind, 0, len(xs) - 1)
        self.update()

    def onpick(self, event):

        if event.artist != line:
            return True

        N = len(event.ind)
        if not N:
            return True

        # the click locations
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata

        distances = np.hypot(x - xs[event.ind], y - ys[event.ind])
        indmin = distances.argmin()
        dataind = event.ind[indmin]

        self.lastind = dataind
        self.update()

    def update(self):
        if self.lastind is None:
            return

        dataind = self.lastind

        ax2.cla()
        ax2.plot(X[dataind])

        ax2.text(0.05, 0.9, 'mu=%1.3f\nsigma=%1.3f' % (xs[dataind], ys[dataind]),
                 transform=ax2.transAxes, va='top')
        ax2.set_ylim(-0.5, 1.5)
        self.selected.set_visible(True)
        self.selected.set_data(xs[dataind], ys[dataind])

        self.text.set_text('selected: %d' % dataind)
        fig.canvas.draw()
def MaxFiltPeaks2d(im):
    """Find peaks in 2D using Max Filter"""
    import scipy
    import scipy.ndimage as ndimage
    import scipy.ndimage.filters as filters

    neighborhood_size = 7
    threshold = 65

    data_max = filters.maximum_filter(im, neighborhood_size)
    maxima = (im == im_max)
    data_min = filters.minimum_filter(im, neighborhood_size)
    #diff = ((data_max - data_min) > threshold)
    diff = ((im_max) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2
        y.append(y_center)

    return [x,y]
