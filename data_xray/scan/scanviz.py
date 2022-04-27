from ..modules import *

#####SXM IMAGE FILES

##this is the new version working with xarray dataset
#from plotly import tools
#from tqdm import tqdm

#list of functions for image analysis

class ScanViz(object):
    def __init__(self, scan_object):
        try:
            self.scan = scan_object
        except:
            print("can't find xarrayed image")
            return
    
    def get_image(self, chan, backw=0):
    
        fb = 'backward' if backw else 'forward'
        self.fb = fb

        im2 = self.scan.signals[chan][fb]
        if np.any(np.isnan(im2)):
            cleanrange = np.setxor1d(np.arange(np.size(im2, 0)), np.unique(np.where(np.isnan(im2))[0]))
            im2 = im2[cleanrange]

        self.original_size = im2.shape

        im2 = (im2 - np.min(im2))# / (np.max(im2) - np.min(im2))
        im2 = im2/np.max(im2)

        if backw:    
            im2 = np.fliplr(im2)

        if self.scan.header['scan_dir'] == 'up':
            im2 = np.flipud(im2)   
        
        self.original_image = im2
        self.image = im2
        self.chan = chan
        return self
    

    def crop(self, cropped):
        #list needs to be [ymin,ymax,xmin,xmax]
        self.image = self.image[cropped[0]:cropped[1], cropped[2]:cropped[3]]
        return self

    def high_pass(self, high_pass):
        #list needs to be [ymin,ymax,xmin,xmax]
        
        im2 = HighLowPass2d(self.image.shape, type='high', pxwidth=high_pass)
        crop = 2*high_pass
        self.image = im2[crop:-crop,crop:-crop] #crop the edges
        return self

    def line_subtract(self):
        self.image = SubtractLine(self.image)
        return self
    
    def zoom(self, zoom):
        self.image = ndimage.zoom(im2,zoom=zoom,order=3)
        return self
        
    def plot_image(self, ax, cm='magma'):

        im2_plot = ax.imshow(self.image, cmap=pplt.Colormap(cm), robust=True);

        ax.colorbar(im2_plot,loc='b')
        scale_unit = np.min([i / j for i, j in zip(self.scan.header['scan_range'], self.original_size)])
        sbar = ScaleBar(scale_unit, location='upper right', font_properties={'size': 8})
        ax.add_artist(sbar)
        ax.set_title(self.chan+' '+ self.fb, fontsize=8)
        ax.axis('off')

        return self
        

def AxesColorLimits(arr2d):
    return [np.mean(arr2d) - 3*np.std(arr2d/np.max(arr2d)),np.mean(arr2d) + 3*np.std(arr2d/np.max(arr2d))]

#high pass
def HighLowPass2d(im, type='low', pxwidth=3):
    #this works on numpy arrays and should also work on xarrayed images
    from scipy import ndimage
    lowpass = ndimage.gaussian_filter(im, pxwidth)

    if type == 'high':
        return im - lowpass
    else:
        highp = im-lowpass
        return im - highp

def SubtractLine(im2, deg=2):
    
    (s1,s2) = im2.shape
    Y = im2/1e-9

    X1, X2 = np.mgrid[:s1, :s2]
    X = np.hstack((np.reshape(X1, (s1*s2, 1)) , np.reshape(X2, (s1*s2, 1))))
    X = np.hstack((np.ones((s1*s2, 1)), X))

    YY = np.reshape(Y, (s1*s2, 1))
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)
    plane = np.reshape(np.dot(X, theta), (s1, s2));

    return Y - plane

    #Y_sub = Y - plane

    #im2, _ = spiepy.flatten_poly_xy(im2/1e-9, deg=deg)
    

def PlotImage(_Scan, chan, ax, backw=0, cm='magma',zoom=1, high_pass=None, cropped=None):
    
    #from matplotlib.colors import LogNorm
    
    fb = 'backward' if backw else 'forward'
    im2 = _Scan.signals[chan][fb]
    if np.any(np.isnan(im2)):
        cleanrange = np.setxor1d(np.arange(np.size(im2, 0)), np.unique(np.where(np.isnan(im2))[0]))
        im2 = im2[cleanrange]

    original_size = im2.shape

    if cropped is None:
        im2 = im2      
    else:
        #list needs to be [ymin,ymax,xmin,xmax]
        im2 = im2[cropped[0]:cropped[1], cropped[2]:cropped[3]]
  
    if chan.lower()=='z':
        try:
            if high_pass is not None:
                im2 = HighLowPass2d(im2, type='high', pxwidth=high_pass)
                crop = 2*high_pass
                im2 = im2[crop:-crop,crop:-crop] #crop the edges
            else:
                im2 = SubtractLine(im2)
        except:
            print('problems subtracting background in ' + _Scan.fname)
            im2 = im2
    #scale = _Scan.header['scan_range']/_Scan.header['scan_pixels']
    #stat = [np.mean(im2), np.mean([np.std(im2 / np.max(im2)), np.std(im2 / np.min(im2))])]
    #color_range = [stat[0] - 3 * stat[1], stat[0] + 3 * stat[1]]
    #im2 = (im2 - np.min(im2))/(np.max(im2)-np.min(im2))
    im2 = (im2 - np.min(im2))# / (np.max(im2) - np.min(im2))
    im2 = im2/np.max(im2)

    if zoom != 1:
         im2 = ndimage.zoom(im2,zoom=zoom,order=3)
    

    if backw:    
        im2 = np.fliplr(im2)
        
    if _Scan.ds.attrs['scan_dir'] == 'up':
        im2 = np.flipud(im2)

        #ax.imshow(np.fliplr(im2), cmap=cm, norm=LogNorm(vmin=0.05, vmax=1.1));

        #im2_plot = ax.imshow(np.fliplr(im2), cmap=pplt.Colormap(cm), robust=True)
        
        #ax.imshow(np.fliplr(im2), cmap=cm, norm=LogNorm(vmin=0.05, vmax=1.1));
        
        
    im2_plot = ax.imshow(im2, cmap=pplt.Colormap(cm), robust=True);

    ax.colorbar(im2_plot,loc='b')
    scale_unit = np.min([i / j for i, j in zip(_Scan.header['scan_range'], original_size)])
    sbar = ScaleBar(scale_unit, location='upper right', font_properties={'size': 8})
    ax.add_artist(sbar)
    ax.set_title(chan+' '+ fb, fontsize=8)
    ax.axis('off')
    
    #fp = sbar.get_font_properties()
    #fp.set_size(8)
    #ax.axes([0.08, 0.08, 0.94 - 0.08, 0.94 - 0.08])  # [left, bottom, width, height]
    #ax.axis('scaled')
    # ax.set_title(phdf+j)
    # ax.set_title('$'+j+'$')
    return im2


def PlotScan(_Scan, chans='all', scanid=0, scandir=2, zoom=1, high_pass=None):
    #scandir 2 for forward-backward, 1 for forward only
    # import plotly.tools as tls
    #     matplotlib.rcParams['font.size'] = 8
    #     matplotlib.rcParams['font.family'] = ['sans-serif']
    #     # chans two values 'fil' or 'all'. All - just plot all channels
    #     # ppl - means use pyplot interface
    #     # 'fil' - plot ony those with meaningful std (aka data). Std threshdol 1e-2
    #     # the plotter! plots all channels contained in the sxm-dict structure

    plotted = []
    fn = _Scan.fname

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

    fig3 = plt.figure(figsize=(3*(1+len(chans)),8))
    gs = gridspec.GridSpec(scandir, len(chans))

    for j, ch in enumerate(chans):


        for scand in np.arange(scandir):
            ax3 = plt.subplot(gs[scand,j])
            p2 = PlotImage(_Scan, chan=ch, ax=ax3, backw=scand, zoom=zoom, high_pass=high_pass)
            plotted.append(p2)

    plt.subplots_adjust(left=0.11, bottom=0.01, right=0.9, top=0.93, wspace=0.54, hspace=0.1)
    plt.tight_layout(pad=0)

    fig3.suptitle("Scan #"+ str(scanid) + ':' + fn, size=12, y=1.02)

    def onresize(event):
        plt.tight_layout()

    cid = fig3.canvas.mpl_connect('resize_event', onresize)

    return fig3, plotted

