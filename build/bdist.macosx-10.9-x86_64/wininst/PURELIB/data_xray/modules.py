#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 18:31:02 2017

@author: peter
"""

import numpy as np
import deepdish as dd
from copy import copy

import pandas as pd
import seaborn as sns
import inspect
import re
import os
import sys
import copy
import subprocess
#import pyperclip
import base64
import datetime


from tqdm import tqdm
from time import time
from struct import unpack
from sklearn.cluster import KMeans
import pandas as pd

from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage import exposure
from sklearn.neighbors import NearestNeighbors
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

#
from yattag import Doc
from yattag import Doc
#from pyearth import Earth



from pptx import Presentation
from pptx.util import Inches

#
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy import misc
from scipy import optimize
from scipy import constants as C
#
#




import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

import fnmatch
import pyperclip
import xarray as xry


#import spiepy
from matplotlib_scalebar.scalebar import ScaleBar
