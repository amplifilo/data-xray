from data_xray.modules import *
from .localmodules import *

#this is feature detection in 2D. A bit more evolved than 1D at this point
import numpy as np

class Filter():
    def __init__(self, im=None):
        self.im = im

    @staticmethod
    def fft2d(im):
        """Retun 2D FFT amplitude"""
        f = np.fft.fft2(im*Filter.hamming_window_2d(im))
        fshift = np.fft.fftshift(f)
        return 20*np.log(np.abs(fshift))
 
    @staticmethod
    def hamming_window_2d(im):
        """Hamming Window for arbitrary aspect ratio"""
        hx = np.hamming(im.shape[0])
        hy = np.hamming(im.shape[1])
        return np.sqrt(np.outer(hx,hy))

    @staticmethod
    def bin_erode_peaks_2d(im):

        """
        Takes an image and detect the peaks usingthe local maximum filter.
        Returns a boolean mask of the peaks (i.e. 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise)
        """


        # define an 8-connected neighborhood
        neighborhood = generate_binary_structure(2,2)

        #apply the local maximum filter; all pixel of maximal value
        #in their neighborhood are set to 1
        local_max = maximum_filter(im, footprint=neighborhood)==im
        #local_max is a mask that contains the peaks we are
        #looking for, but also the background.
        #In order to isolate the peaks we must remove the background from the mask.

        #we create the mask of the background
        background = (im==0)

        #a little technicality: we must erode the background in order to
        #successfully subtract it form local_max, otherwise a line will
        #appear along the background border (artifact of the local maximum filter)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

        #we obtain the final mask, containing only peaks,
        #by removing the background from the local_max mask
        detected_peaks = local_max - eroded_background

        return detected_peaks

    @staticmethod
    def image_threshold(im):
        """threshold filter: im is an image"""

        try_all_threshold(im, figsize=(10, 8), verbose=False)
        plt.show()

    @staticmethod
    def image_threshold_otsu(im, plotit=0):
        """threshold Otsu"""

        thresh = threshold_otsu(im) * 24
        binary = im < thresh
        im_binary = im * binary
        if plotit:
            fig2, ax2 = plt.subplots(1, 1)
            img2 = ax2.imshow(-im_binary, cmap=plt.cm.gray)
            plt.colorbar(img2)
            plt.show()
        return im_binary

    @staticmethod
    def image_distance_transform(im_binary):
        """Image Distance Transform"""

        distance = ndi.distance_transform_edt(-im_binary)
        local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)))
        markers = ndi.label(local_maxi)[0]
        labels = watershed(-distance, markers, mask=imb)
        return labels


class BlobFinder():

    def __init__(self):
        pass

    @staticmethod
    def image_find_blobs(im, max_sigma, min_sigma, threshold):
        """
        alternative
        blobs_log = blob_log(im4, max_sigma=7, num_sigma=2, threshold=2)
        Compute radii in the 3rd column.
        blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

        pretty good particle finder using blob_dog function from scikit

        """
        try:
            blobs_dog = blob_dog(im, max_sigma=max_sigma, min_sigma=min_sigma, threshold=threshold)

            if len(blobs_dog):
                blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)
                sz = np.asarray([[i[2], im[int(i[0]),int(i[1])]] for i in blobs_dog])
                #nclust = kmeans_core(sz,2)

                fig,ax = plt.subplots(1, 1, figsize=(6,6), dpi=100)
                climit = [np.median(im) - 3*np.std(im), np.median(im) + 3*np.std(im)]
                ax.imshow(im, cmap = plt.cm.gray, interpolation='nearest', clim=climit )

                for j in np.arange(blobs_dog.shape[0]):
                #if nclust[j] == 1:
                    y, x, r = blobs_dog[j]
                    ax.plot(x,y,'r.')
                ax.axis('tight')
                ax.set_aspect('equal')

            else:
                print('check params')

            #return blobs_dog
        except ValueError:
            print("need to check those params")

    @staticmethod
    def get_blob_props(im, blobs_dog):
        """
        read-out function for the blob properties
        :param im:
        :param blobs_dog:
        :return:
        """

        try:
            params = list(blobs_dog.widget.kwargs.values())[1:]
            return blob_dog(im, max_sigma=params[0], min_sigma=params[1], threshold=params[2])
        except ValueError:
            print("need to check those params")


def VoronoiTesselation(pts, radius=None):
    """
    #Voronoi tesselation
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions.

        Parameters
        ----------
        vor : Voronoi
            Input diagram
        radius : float, optional
            Distance to 'points at infinity'.

        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.

        """
    vor = Voronoi(pts)

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all([v >= 0 for v in vertices]):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

    from sklearn import metrics
    from sklearn.metrics import pairwise_distances
    from sklearn.cluster import KMeans

def PlotVoronoi(im4, pts, plotregions='all'):

    """
    overlay Voronoi tesselation over the image

    :param plotregions 'all' if you want to see them all, list if you want to downselect some
    :param im4: image array
    :param pts: lattice of object centers
    :return: figure that overlays voronoi tesselation over the image
    """

    def poly_center(polygon):
        p2 = path.Path(polygon)
        x, y = p2.vertices[:, 0], p2.vertices[:, 1]
        return (sum(x) / len(p2.vertices), sum(y) / len(p2.vertices))


    regions, vertices = VoronoiTesselation(pts)

    if not(isinstance(plotregions,list)):
        plotregions = list(np.arange(len(regions)))

    fig3, ax3 = plt.subplots(1, 1)
    ax3.imshow(np.exp(im4), alpha=0.7, cmap=plt.cm.gray)#, clim=[np.min(im4) * 0.1, np.max(im4) * 1.1])
    print(plotregions)
    for ir, j in enumerate(regions):

        if ir in plotregions:

            plt.fill(*zip(*vertices[j]), facecolor='none', edgecolor='black', alpha=1.0)
            plt.annotate(s=str(ir), xy=poly_center(vertices[j]), horizontalalignment='center')
    plt.xlim([0, np.size(im4, 0)])
    plt.ylim([np.size(im4, 1), 0])


    #scalebar = ScaleBar(scale_unit)
    #ax3.add_artist(scalebar)

    ax3.axis('off')
    plt.tight_layout()
    return fig3
    #fig3.savefig('phase2_1_transition.svg', type='svg')


def PolygonMaskExtract(im4, polygon, plotit=0):

    """
    extract polygon mask
    for VoronoiTesselation, polygon will be vertices[regions[j]]
    :return:
    polygon = vertices[regions[2]]
    """

    p = path.Path(polygon)

    nr, nc = im4.shape
    # im4 = np.flipud(im4)

    ygrid, xgrid = np.mgrid[:nr, :nc]
    xypix = np.vstack((xgrid.ravel(), ygrid.ravel())).T

    mask = p.contains_points(xypix)
    mask = mask.reshape(im4.shape)

    masked = np.ma.masked_array(im4, ~mask)

    xmin, xmax = int(polygon[:, 0].min()), int(np.ceil(polygon[:, 0].max()))
    ymin, ymax = int(polygon[:, 1].min()), int(np.ceil(polygon[:, 1].max()))
    trimmed = masked[ymin:ymax, xmin:xmax]

    if plotit:
        fig, ax = plt.subplots(2, 2)

        ax[0, 0].imshow(im4, cmap=plt.cm.gray)
        ax[0, 0].set_title('original')
        ax[0, 1].imshow(mask, cmap=plt.cm.gray)
        ax[0, 1].set_title('mask')
        ax[1, 0].imshow(masked, cmap=plt.cm.gray)
        ax[1, 0].set_title('masked original')
        ax[1, 1].imshow(trimmed, cmap=plt.cm.gray)
        ax[1, 1].set_title('trimmed original')

        plt.show()
    return masked, trimmed

def VoronoiPolygonAreas(pts):
    """
    calculate areas of Voronoi polygons
    """

    def PolyArea(x,y):
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

    regions, vertices = VoronoiTesselation(pts)
    allareas = []
    for j in range(len(regions)):
        allareas.append(PolyArea(vertices[regions[j]][:,0], vertices[regions[j]][:,1]))

    return allareas

def VoronoiAreasKmeans(pts, k, plotit=0, iters=0, area_cutoff = np.inf):
    """
    kmeans clustering of Voronoi polygons (or any other areas). So far by area size
    which may not be the best choice for some cases

    :param pts:
    :param k:
    :param plotit:
    :param iters:
    :param area_cutoff:
    :return:
    """



    def fK(X, thisk, Skm1=0):
        Nd = len(X[0])
        a = lambda k, Nd: 1 - 3/(4*Nd) if k == 2 else a(k-1, Nd) + (1-a(k-1, Nd))/6
        clusters = dict()
        km = KMeans(n_clusters=thisk, random_state=1).fit(X)
        mu = km.cluster_centers_
        for j in range(thisk):
            clusters[j] = [X[i] for i in np.where(km.labels_==0)[0]]

        Sk = sum([np.linalg.norm(mu[i]-c)**2 for i in range(thisk) for c in clusters[i]])
        if thisk == 1:
            fs = 1
        elif Skm1 == 0:
            fs = 1
        else:
            fs = Sk/(a(thisk,Nd)*Skm1)
        return fs, Sk
    def fK_k(X, kmax=11, plotax=None):
        optk = list()
        #firstk = fK(X,1)
        optk.append(fK(X,1))
        for j in range(kmax):
            optk.append(fK(X,thisk=j+1,Skm1=optk[j][1]))
        optk = optk[1:]
        if type(plotax)==None:
            fig1,plotax = plt.subplots(1,1)

        plotax.plot(range(kmax), [i[0] for i in optk],'g.-')
        plotax.set_ylim([0,np.max([i[0] for i in optk])*1.1])

    allareas = VoronoiPolygonAreas(pts)
    allareas = [i for i in allareas if i<area_cutoff]
    X = np.asarray(allareas).reshape(-1,1)
    
    km = KMeans(n_clusters=k, random_state=1).fit(X)
    labels = km.labels_
    centroids = list()
    for j in range(k):
        #sarr = [allareas[i] for i in np.where(labels==j)[0]]
        centroids.append([km.cluster_centers_[j], len(np.where(labels==j)[0])])

    centroids =  np.asarray(centroids)
    centroids = centroids[centroids[:,0].argsort()]

    if plotit:
        fig1, ax1 = plt.subplots(1,2)
        ax1[0].loglog(centroids[:,0],centroids[:,1],'r.--')
        fK_k(X,plotax=ax1[1])

    return km

