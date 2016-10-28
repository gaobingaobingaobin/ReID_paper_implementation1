# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:05:39 2016

@author: 
"""

import h5py
import numpy as np

f = h5py.File('./CUHK03/cuhk-03.mat')
'''
np.swapaxis(np.array(f[f[f['labeled'][0][i]][j][k]]),0,2)
this expression will get a numpy array of a picture with axis order is 'tf'

f[f['labeled'][0][i]][0].size
this expression return the numbers of id captured by i pari of cameras

i: from 0-4, the number i pair of cameras
j: from 0-9, the number j pictures of identity k captured by i pair of cameras
k: identity numbers
'''
