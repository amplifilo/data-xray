#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 10:37:38 2016

@author: peter
"""
from .modules import *
import pandas as pd

def p_or(vvec,but=[0,0]):
    b1 = vvec < but[0]
    b2 = vvec > but[1]
    return b1 | b2

def p_and(vvec,but=[0,0]):
    b1 = vvec > but[0]
    b2 = vvec < but[1]
    return b1 & b2

def cart2pol(x, y, xc=0, yc=0):
    rho, phi = list(),list()
    for xi,yi in zip(x,y):
        xi -= xc
        yi -= yc
        rho.append(np.sqrt(xi**2 + yi**2))
        phi.append(np.arctan2(yi, xi))
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def euclidean_distance(x,y):
#simple euclidean distance
    return np.sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))


def inlist(self, lst, keyword):
#locate items in list
    ind = []
    for j in lst:
        if bool(re.search(keyword, j, re.IGNORECASE)):
            ind.append(j)
    return ind

def split_into_groups(lst, pergroup=3):
    import itertools
    return list(itertools.zip_longest(*(iter(lst),) * pergroup))

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def npa(x):
    return np.asarray([j for i in x for j in i]) #flatten list into numpy array via list comprehension


def nodup(seq):
    # order preserving
        noDupes = []
        [noDupes.append(i) for i in seq if not noDupes.count(i)]
        return noDupes


def idt(tdict, index = 0):
     return tdict[list(tdict.keys())[index]]

def strIntersection(str1, str2):
    for i in str1:
        str3 = ''
        str3 = str3 = str3.join(i for i in str1 if i in str2 and i not in str3)
    return str3

def strSetXor(str1, str2):
    for i in str1:
        str3 = ''
        str3 = str3.join(i for i in str1 if i not in str2)
    return str3

def darkPlot():
    sns.set(style="ticks", context="talk")
    plt.style.use("dark_background")


nonecheck = lambda x: '' if x is None else x
labelcheck = lambda d,x: d[x] if x in d else ''
nearest = lambda d,x: np.argmin(np.abs(d - x))

class Container(object):
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
    def _from_dict(self, d):
        for k in list(d.keys()):
                setattr(self, k, d[k])
    def addxy(self,x,y):
        setattr(self, 'x', x)
        setattr(self, 'y', y)
################

