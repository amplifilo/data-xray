"""
Created on Wed Jan 11 10:50:02 2017

@author: 5nm
"""

from ..modules import *
from sklearn.decomposition import PCA



def KmeansIterative(X=None, keach=3, iters=2):
    """
    Rather experimental nestes k-means. Recommend using dengdrogram or hierarchical clustering instead
        :param X= Data matrix: 
        :param keach= 3 clusters each: 
        :param iters= defauklt two iterations: 
    """
    
    from sklearn import metrics
    from sklearn.metrics import pairwise_distances
    #iters = 3
    #keach = 3
    for j in range(iters):
        km = KMeans(n_clusters=keach, random_state=1).fit(X)
        labels = km.labels_
        pops = np.asarray([[m,len(np.where(km.labels_==m)[0])] for m in range(keach)])
        subind = np.where(km.labels_ == pops[pops[:,1].argsort()][-1][0])[0]
        X = X[subind]
    print(len(X))
    return X


def HirearchicClustering(X=None, k=None, ax=None):
    """
    Basic hierarchical clustering

        :param X=data array: 
        :param k=number of desired clusters: 
        :param plotit=0: 
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.cluster.hierarchy import fcluster

    xy,z = np.product(X.shape[:-1]),X.shape[-1]
    dat2 = X.reshape((xy,z))
    Z = linkage(dat2, 'ward')
    cmask = fcluster(Z, k, criterion='maxclust')

    #xsel,ysel= np.where(cmask.reshape(20,20)==1)
    cmask_square = cmask.reshape(X.shape[:-1])-1

    if ax is None:
        return cmask_square
    else:
        return dendrogram(Z, leaf_rotation=90, leaf_font_size=8, truncate_mode='lastp', p=k, ax=ax), cmask_square
    
        
    #

def ScpDendrogram(X=None, lastp=10):
    """
    adopted from Joern Hees blog, scipy flavor of dendrogram
        :param X=data array: 
        :param lastp=max level of the dendrogram: 
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist

    def fancy_dendrogram(*args, **kwargs):
        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        annotate_above = kwargs.pop('annotate_above', 0)

        ddata = dendrogram(*args, **kwargs)

        if not kwargs.get('no_plot', False):
            plt.title('Hierarchical Clustering Dendrogram (truncated)')
            plt.xlabel('sample index or (cluster size)')
            plt.ylabel('distance')
            for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    plt.plot(x, y, 'o', c=c)
                    plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                 textcoords='offset points',
                                 va='top', ha='center')
            if max_d:
                plt.axhline(y=max_d, c='k')
        return ddata


    Z = linkage(X, 'ward')
    c, coph_dists = cophenet(Z, pdist(X))

    # calculate full dendrogram
    fancy_dendrogram(
        Z,
        truncate_mode='lastp',
        p=12,
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
        annotate_above=10,  # useful in small plots so annotations don't overlap
    )
    plt.show()

def PcaCore(X=None, dim=2, figs=[]):
    """
    Handler for principal component analysis

        :param X=None: 
        :param dim=2: 
        :param figs=[]: 
    """
    from sklearn.decomposition import PCA
    from matplotlib import gridspec

    pca = PCA(whiten=False).fit(X)

    if len(figs) >0:
        
        figs[0].suptitle('PCA components', size=11)
        G = gridspec.GridSpec(3, 3)
        for j in range(9):
            ax = figs[0].add_subplot(G[j])
            ax.plot(pca.components_[j])
            ax.set_title('comp #' + str(j), size=10)
    if len(figs) > 1:
        _ax = figs[1].add_subplot(111)
        _ax.semilogy(np.cumsum(pca.explained_variance_ratio_))
        _ax.set_title('score plot')
    load_maps = pca.transform(X)

    if len(figs)>2 and dim > 1:
        figs[2] = plt.figure()
        figs[2].suptitle('loading maps', size=11)
        G = gridspec.GridSpec(3, 3)
        for j in range(9):
            ax = figs[2].add_subplot(G[j])

            side = int(np.sqrt(load_maps.shape[0]))
            ax.imshow(load_maps[:side**2,j].reshape(side,side), cmap=plt.cm.RdBu)
            ax.set_title('map #' + str(j), size=10)
    
    return {'load_maps':load_maps, 'eigenims':pca.components_, 'weight':pca.explained_variance_ratio_, 'pca':pca}

def KmeansCore(X=None, nclust=None):

    km = KMeans(nclust)
    km.fit(X)
    # clustind = km.predict(dat)
    return km  # clustind

def PcaDenoise(X=None, keep_components=3, savgolparms=None):
    """
    PCA denoising of the array
        :param X=data array. Dimensions (x), number of samples (y)
        :param keep_components=3: 
        :param savgolparms=None: 
    """
    import scipy.signal as scp
    
    pcadict = PcaCore(X=X)
    pca = pcadict['pca']

    load_maps = pca.fit_transform(X)
    if savgolparms is not None:
        for j in range(pca.components_.shape[0]):
              pca.components_[j] = scp.savgol_filter(pca.components_[j], window_length=savgolparms[0], polyorder=savgolparms[1])
    #load_maps = pca.transform(X)
    load_maps[:,keep_components:] = 0
    return pca.inverse_transform(load_maps)


def PcaKmeans(X=None, dims=None, nclust=3, pf=lambda x, i: x, fig=None):
    """
    docstring here
        :param X=data array: 
        :param dims=PCA components to keep: 
        :param nclust=k-means clusters to select: 
        :param pf=modification function: 
        :param ax= supply array of two axes for plotting: 
    """ 
    from mpl_toolkits.mplot3d import Axes3D


    if dims is None:
        pca = PCA()
        pca.fit(X)
    else:
        pcadict = PcaCore(X=X, dim=dims, figs=[])
        pca = pcadict['pca']

    
    Y = pca.fit_transform(X)
    km = KMeans(n_clusters=nclust, random_state=0).fit(Y[:, 0:nclust])

    if fig is not None:
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection='3d')

        clrs = ['b.-', 'g.-', 'r.-']

        ax1.plot(np.arange(20), 100*np.cumsum(pca.explained_variance_ratio_[:20]), 'b.-')
        ax1.set_title('PCA variance')
        ax1.set_ylabel('%')
        ax1.set_xlabel('PCA component')

        # ax[1].plot(Y[km.labels_ == 0, 0], Y[km.labels_ == 0, 1], 'r.')
        # ax[1].plot(Y[km.labels_ == 1, 0], Y[km.labels_ == 1, 1], 'b.')
        # ax[1].plot(Y[km.labels_ == 2, 0], Y[km.labels_ == 2, 1], 'g.')
        # ax[1].set_xlabel('PCA1 score')
        # ax[1].set_ylabel('PCA2 score')

        ax2.scatter(Y[km.labels_ == 0, 0], Y[km.labels_ == 0, 1], Y[km.labels_ == 0, 2], 'r.')
        ax2.scatter(Y[km.labels_ == 1, 0], Y[km.labels_ == 1, 1], Y[km.labels_ == 1, 2], 'b.')
        ax2.scatter(Y[km.labels_ == 2, 0], Y[km.labels_ == 2, 1], Y[km.labels_ == 2, 2], 'g.')
        ax2.set_xlabel('PCA1')
        ax2.set_ylabel('PCA2')
        ax2.set_zlabel('PCA3')


        plt.legend()
        ax2.axis('tight')
        ax2.dist = 12
        plt.tight_layout()
        #except:
        #    print('please specify axes for pca plots correctly')

    return km, Y

def PcaDbscan(X=None, dims=None, nclust=3, pf=lambda x, i: x, ax=[]):
   

    """
    PCA followed by DBSCAN
        :param X=data array: 
        :param dims=PCA components to keep: 
        :param nclust=k-means clusters to select: 
        :param pf=modification function: 
        :param ax= supply array of two axes for plotting: 
    """ 

    if dims is None:
        pca = PCA()
        pca.fit(X)
    else:
        pcadict = PcaCore(X=X, dim=dims, figs=[])
        pca = pcadict['pca']


    
    Y = pca.fit_transform(X)
    km = KMeans(n_clusters=nclust, random_state=0).fit(Y[:, 0:nclust])

    if len(ax)>0:
        
        clrs = ['b.-', 'g.-', 'r.-']

        ax_3[0].semilogy(np.arange(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, 'b.-')
        ax_3[1].plot(Y[km.labels_ == 0, 0], Y[km.labels_ == 0, 1], 'r.')
        ax_3[1].plot(Y[km.labels_ == 1, 0], Y[km.labels_ == 1, 1], 'b.')
        ax_3[1].plot(Y[km.labels_ == 2, 0], Y[km.labels_ == 2, 1], 'g.')

        fig_4, ax_4 = plt.subplots(1, 1)
        for i in range(nclust):
            for pl_i in np.random.choice(np.where(km.labels_ == i)[0],5):
                ax_4.plot(pf(dat[pl_i], i))

       
    return km

def PcaAffinity(X=None, dims=None, nclust=3, plotit=0, pf=lambda x, i: x, ax=[]):
    """
    PCA followed by Affinity propagration
        :param X=data array: 
        :param dims=PCA components to keep: 
        :param nclust=k-means clusters to select: 
        :param pf=modification function: 
        :param ax= supply array of two axes for plotting: 
    """ 
    
    
    from sklearn.cluster import AffinityPropagation

    if dims is None:
        pca = PCA()
        pca.fit(X)
    else:
        pcadict = PcaCore(X=X, dim=dims, figs=[])
        pca = pcadict['pca']


    
    Y = pca.fit_transform(X)
    # km = KMeans(n_clusters=nclust, random_state=0).fit(Y[:,0:nclust])

    af = AffinityPropagation(preference=-50).fit(Y[:, 0:nclust])
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    n_clusters_ = len(cluster_centers_indices)

    if len(ax)>1:
        fig_3, ax_3 = plt.subplots(1, 2)
        clrs = ['b.-', 'g.-', 'r.-']

        ax_3[0].semilogy(np.arange(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, 'b.-')
        ax_3[1].plot(Y[af.labels_ == 0, 0], Y[km.labels_ == 0, 1], 'r.')
        ax_3[1].plot(Y[af.labels_ == 1, 0], Y[km.labels_ == 1, 1], 'b.')
        ax_3[1].plot(Y[af.labels_ == 2, 0], Y[km.labels_ == 2, 1], 'g.')

        plt.show()
    return af


