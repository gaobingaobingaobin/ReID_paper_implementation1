# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 16:56:56 2016

@author: dingning
"""

import h5py
import numpy as np
from keras.preprocessing import image
from keras.utils import np_utils

'''
np.array(f[f[f['labeled'][0][i]][j][k]]).transpose(2,1,0)
this expression will get a numpy array of a picture with axis order is 'tf'
f[f['labeled'][0][i]][0].size
this expression return the numbers of id captured by i pari of cameras
i: from 0-4, the number i pair of cameras
j: from 0-9, the number j pictures of identity k captured by i pair of cameras
k: identity numbers
'''

class ImageDataGenerator_for_multiinput(image.ImageDataGenerator):
    def fit(self, X,
            augment=False,
            rounds=1,
            seed=None):
        '''
        # Arguments
            X: Numpy array, the data to fit on.
            augment: whether to fit on randomly augmented samples
            rounds: if `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        '''
        if seed is not None:
            np.random.seed(seed)

        X = np.copy(X)
        if augment:
            aX = np.zeros(tuple([rounds * X.shape[0]] + list(X.shape)[1:]))
            for r in range(rounds):
                print 'rounds:',rounds
                for i in range(X.shape[0]):
                    aX[i + r * X.shape[0]] = self.random_transform(X[i])
            X = aX

        return X


def load_positive_data(file_path = './CUHK03/cuhk-03.mat'):
    f = h5py.File(file_path)
    validation_set_index = np.array(f[f['testsets'][0][0]]).T
    test_set_index = np.array(f[f['testsets'][0][1]]).T
    image_array_list_a = []
    image_array_list_b = []
    count = 0
    for i in xrange(5):
        for k in xrange(f[f['labeled'][0][i]][0].size):
            if [i,k] in validation_set_index:
                continue
            elif [i,k] in test_set_index:
                continue
            for ja in xrange(5):                
                a = np.array(f[f[f['labeled'][0][i]][ja][k]])
                if a.size < 3: 
                    continue
                else:
                    for jb in xrange(5,10):
                        b = np.array(f[f[f['labeled'][0][i]][jb][k]])
                        if b.size < 3:
                            continue
                        else:
                            print 'a.shape:',a.shape
                            print 'b.shape:',b.shape
                            a = _resize_image(a.transpose(2,1,0))
                            b = _resize_image(b.transpose(2,1,0))
                            image_array_list_a.append(a)
                            image_array_list_b.append(b)
                            count += 1
            print 'already load',count,'positive pairs'
                
    x_positive_a = np.array(image_array_list_a)
    x_positive_b = np.array(image_array_list_b)
    y_positive = np.ones(len(x_positive_a))
    return [x_positive_a,x_positive_b],y_positive


def load_negative_data(positive_data_length, ratio = 2, file_path = './CUHK03/cuhk-03.mat'):
    f = h5py.File(file_path)
    image_array_list_a = []
    image_array_list_b = []
    print 'total number:',positive_data_length * ratio
    for index in xrange(positive_data_length * ratio):
        if (index+1) % 10000 == 0:
            print 'already loaded:',index+1
        a,b = _random_choose(f)
        a = _resize_image(a.transpose(2,1,0))
        b = _resize_image(b.transpose(2,1,0))
        image_array_list_a.append(a)
        image_array_list_b.append(b)
    x_negative_a = np.array(image_array_list_a)
    x_negative_b = np.array(image_array_list_b)
    y_negative = np.zeros(len(x_negative_a))
    return [x_negative_a,x_negative_b],y_negative


def _resize_image(im_array,shape=(160,60)):
    if im_array.shape[2] > 3:
        im_array = im_array.transpose(2,1,0)
    print im_array.shape
    im = image.array_to_img(im_array,dim_ordering = 'tf')
    im = im.resize(shape)
    array = image.img_to_array(im,dim_ordering = 'tf')
    return array.transpose(1,0,2)


def _random_choose(f):
    validation_set_index = np.array(f[f['testsets'][0][0]]).T
    test_set_index = np.array(f[f['testsets'][0][1]]).T
    while True:
        i = np.random.randint(0,5)
        ja = np.random.randint(0,5)
        jb = np.random.randint(5,10)
        ka = np.random.randint(0,f[f['labeled'][0][i]][0].size)
        while [i,ka] in validation_set_index or [i,ka] in test_set_index:
            ka = np.random.randint(0,f[f['labeled'][0][i]][0].size)
        kb = np.random.randint(0,f[f['labeled'][0][i]][0].size)
        while [i,kb] in validation_set_index or [i,kb] in test_set_index:
            kb = np.random.randint(0,f[f['labeled'][0][i]][0].size)
        while ka == kb:
            kb = np.random.randint(0,f[f['labeled'][0][i]][0].size)
        a = np.array(f[f[f['labeled'][0][i]][ja][ka]])
        b = np.array(f[f[f['labeled'][0][i]][jb][kb]])
        if a.size > 2 and b.size > 2: break
    return a,b
    

def load_validation_data(file_path = './CUHK03/cuhk-03.mat'):
    f = h5py.File(file_path)
    image_array_list_a = []
    image_array_list_b = []
    count = 0
    validation_set_index = np.array(f[f['testsets'][0][0]]).T
    print 'Begin to create positive validation data.'
    for i,k in validation_set_index:
        for ja in xrange(5):
            a = np.array(f[f[f['labeled'][0][i]][ja][k]])
            if a.size < 3: 
                continue
            else:
                for jb in xrange(5,10):
                    b = np.array(f[f[f['labeled'][0][i]][jb][k]])
                    if b.size < 3:
                        continue
                    else:
                        a = _resize_image(a.transpose(2,1,0))
                        b = _resize_image(b.transpose(2,1,0))
                        image_array_list_a.append(a)
                        image_array_list_b.append(b)
                        count += 1
        print 'already made',count,'positive pairs'
    x_positive_a = np.array(image_array_list_a)
    x_positive_b = np.array(image_array_list_b)
    y_positive = np.ones(len(x_positive_a))
    print 'positive validation data numbers:',len(y_positive)
    print 'Begin to make negative validation data.'
    negative_list_a = []
    negative_list_b = []
    count = 0
    for i in xrange(100):
        for j in xrange(i+1,100):
            for ja in xrange(5):
                a = np.array(f[f[f['labeled'][0][validation_set_index[i][0]][ja][validation_set_index[i][1]]]])
                if a.size < 3:
                    continue
                else:
                    for jb in xrange(5,10):
                        b = np.array(f[f[f['labeled'][0][validation_set_index[j][0]][jb][validation_set_index[j][1]]]])
                        if b.size < 3:
                            continue
                        else:
                            a = _resize_image(a.transpose(2,1,0))
                            b = _resize_image(b.transpose(2,1,0))
                            negative_list_a.append(a)
                            negative_list_b.append(b)
                            count += 1
                            if count == 2 * len(y_positive):
                                x_negative_a = np.array(negative_list_a)
                                x_negative_b = np.array(negative_list_b)
                                y_negative = np.zeros(len(x_negative_b))
                                print 'negative validation set done.'
                                print 'negative validation data numbers:',len(y_negative)
                                print 'Begin to store the validation set in local disk.'
                                x_val_a = np.concatenate([x_positive_a,x_negative_a],axis = 0)
                                x_val_b = np.concatenate([x_positive_b,x_negative_b],axis = 0)
                                y_val   = np.concatenate([y_positive,y_negative],axis = 0)
                                x_val_1 = f.create_dataset('x_val_1',data = x_val_a)
                                x_val_2 = f.create_dataset('x_val_2',data = x_val_b)
                                y_val   = f.create_dataset('y_val',  data = y_val)
                                print 'validation set already stored in local disk.'
                                
    
if __name__=='__main__':
    print 'loading positive data......'
    x_pos, y_pos = load_positive_data()
    print 'already loaded all positive data.'
    print 'positive data number:',len(y_pos)
    print 'positive data augmentation begin......'
    Data_Generator = ImageDataGenerator_for_multiinput(
                     width_shift_range=0.05,
                     height_shift_range=0.05)
    print 'for the first input:'
    x_pos[0] = Data_Generator.fit(x_pos[0],augment=True,rounds=5,seed=1217)
    print 'for the second input:'
    x_pos[1] = Data_Generator.fit(x_pos[1],augment=True,rounds=5,seed=1217)
    y_pos = np.repeat(y_pos,5,axis=0)
    print 'positive data augmentation done.'
    print 'positive X data number after augmentation:',len(x_pos[0])
    print 'positive Y data number after augmentation:',len(y_pos)
    print 'loading negative data......'
    x_neg,y_neg = load_negative_data(len(y_pos))
    print 'already loaded negative data.'
    print 'negative data number:',len(y_neg)
    X1 = np.concatenate([x_pos[0],x_neg[0]],axis=0)
    print 'preparation for X1 done. Begin to store in disk as hdf5 file.'
    f = h5py.File('train_data_all.hdf5')
    x_train_1 = f.create_dataset('x_train_1',data = X1, chunks = True)
    print 'already stored in local disk.'
    print 'Begin to store the X2......'
    X2 = np.concatenate([x_pos[1],x_neg[1]],axis=0)
    x_train_2 = f.create_dataset('x_train_2',data = X1, chunks = True)
    print 'already stored in local disk.'
    print 'Begin to store the Y......'
    Y = np.concatenate([y_pos,y_neg],axis=0)
    Y = np_utils.to_categorical(Y, 2)
    y_train = f.create_dataset('y_train',data = Y, chunks = True)
    print 'already stored in local disk.'
    print 'HDF5 data set created done.'
    print 'X train data number:',len(X1[0])
    print 'Y train data number:',len(Y)
    load_validation_data()
