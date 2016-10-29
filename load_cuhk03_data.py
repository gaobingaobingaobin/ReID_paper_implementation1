# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:05:39 2016

@author: 
"""
import h5py
import numpy as np
from keras.preprocessing import image

'''
np.swapaxis(np.array(f[f[f['labeled'][0][i]][j][k]]),0,2)
this expression will get a numpy array of a picture with axis order is 'tf'

f[f['labeled'][0][i]][0].size
this expression return the numbers of id captured by i pari of cameras

i: from 0-4, the number i pair of cameras
j: from 0-9, the number j pictures of identity k captured by i pair of cameras
k: identity numbers
'''

def load_positive_data(file_path = './CUHK03/cuhk-03.mat'):
    f = h5py.File(file_path)
    image_array_list = []
    for i in xrange(5):
        for k in xrange(f[f['labeled'][0][i]][0].size):
            for j in xrange(5):
                a = np.array(f[f[f['labeled'][0][i]][j][k]])
                b = np.array(f[f[f['labeled'][0][i]][j+5][k]])
                if b.size == 1: continue
                a = _resize_image(np.swapaxis(a,0,2))
                b = _resize_image(np.swapaxis(b,0,2))
                image_array_list.append(np.concatenate([a,b],axis=2))
    x_positive = np.array(image_array_list)
    y_positive = np.array(['positive'] * len(image_array_list))
    return x_positive,y_positive


def load_negative_data(positive_data_length, ratio = 2, file_path = './CUHK03/cuhk-03.mat'):
    f = h5py.File(file_path)
    image_array_list = []
    for _ in xrange(positive_data_length * ratio):
        a,b = _random_choose(f)
        a = _resize_image(np.swapaxis(a,0,2))
        b = _resize_image(np.swapaxis(b,0,2))
        image_array_list.append(np.concatenate([a,b],axis=2))
    x_negative = np.array(image_array_list)
    y_negative = np.array(['negative'] * len(image_array_list))
    return x_negative,y_negative


def _resize_image(im_array,shape=(160,60)):
    im = image.array_to_img(im_array,dim_ordering = 'tf')
    im = im.resize(shape)
    array = image.img_to_array(im,dim_ordering = 'tf')
    return array


def _random_choose(f):
    while True:
        i = np.random.randint(0,5)
        j = np.random.randint(0,5)
        k1 = np.random.randint(0,f[f['labeled'][0][i]][0].size)
        k2 = np.random.randint(0,f[f['labeled'][0][i]][0].size)
        while k1 == k2:
            k2 = np.random.randint(0,f[f['labeled'][0][i]][0].size)
        a = np.array(f[f[f['labeled'][0][i]][j][k1]])
        b = np.array(f[f[f['labeled'][0][i]][j+5][k2]])
        if a.size > 1 and b.size > 1: break
    return a,b











