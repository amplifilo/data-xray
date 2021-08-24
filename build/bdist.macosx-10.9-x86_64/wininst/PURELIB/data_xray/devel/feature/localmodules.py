from skimage.filters import threshold_otsu, try_all_threshold
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy import ndimage as ndi
from scipy.spatial import Voronoi, voronoi_plot_2d

from matplotlib import path
