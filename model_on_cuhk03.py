# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 22:20:31 2016
@author: dingning
"""
import numpy as np
np.random.seed(1217)
from keras import backend as K
from keras.models import Model
from keras.layers import Input,Dense,Convolution2D,Activation,MaxPooling2D,Flatten,merge
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.utils import np_utils
from load_cuhk03_data import load_positive_data,load_negative_data
from keras.preprocessing import image as pre_image

K._IMAGE_DIM_ORDERING = 'tf'

class NumpyArrayIterator_for_multiinput(pre_image.Iterator):

    def __init__(self, X, y, image_data_generator=None,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='default'):
        
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.X1 = X[0]
        self.X2 = X[1]
        self.y = y
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        super(NumpyArrayIterator_for_multiinput, self).__init__(X[0].shape[0], batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x1 = np.zeros(tuple([current_batch_size] + list(self.X1.shape)[1:]))
        batch_x2 = np.zeros(tuple([current_batch_size] + list(self.X2.shape)[1:]))
        for i, j in enumerate(index_array):
            x1 = self.X1[j]
            x2 = self.X2[j]
            if self.image_data_generator is None:
                batch_x1[i] = x1
                batch_x2[i] = x2
                continue
            else:
                x1 = self.image_data_generator.random_transform(x1.astype('float32'))
                x2 = self.image_data_generator.random_transform(x2.astype('float32'))
                x1 = self.image_data_generator.standardize(x1)
                x2 = self.image_data_generator.standardize(x2)
                batch_x1[i] = x1
                batch_x2[i] = x2
       
        if self.y is None:
            return [batch_x1, batch_x2]
        batch_y = self.y[index_array]
        return [batch_x1,batch_x2], batch_y

class SGD_new(SGD):
    '''Stochastic gradient descent, with support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    '''
    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0.)
        self.lr = K.variable(lr)
        self.momentum = K.variable(momentum)
        self.decay = K.variable(decay)
        self.inital_decay = decay

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = []

        lr = self.lr
        if self.inital_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations)) ** 0.75
            self.updates .append(K.update_add(self.iterations, 1))

        # momentum
        shapes = [K.get_variable_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates


def concat_iterat(input_tensor):
    input_expand = K.expand_dims(K.expand_dims(input_tensor, -2), -2)
    x_axis = []
    y_axis = []
    for x_i in range(5):
        for y_i in range(5):
            y_axis.append(input_expand)
        x_axis.append(K.concatenate(y_axis, axis=2))
        y_axis = []
    return K.concatenate(x_axis, axis=1)

def cross_input_both(X):
    tensor_left = X[0]
    tensor_right = X[1]
    x_length = K.int_shape(tensor_left)[1]
    y_length = K.int_shape(tensor_left)[2]
    cross_y_left = []
    cross_x_left = []
    cross_y_right = []
    cross_x_right = []
    #scalar = K.ones([5,5])
    tensor_left_padding = K.spatial_2d_padding(tensor_left,padding=(2,2))
    tensor_right_padding = K.spatial_2d_padding(tensor_right,padding=(2,2))
    for i_x in range(2, x_length + 2):
        for i_y in range(2, y_length + 2):
            cross_y_left.append(tensor_left_padding[:,i_x-2:i_x+3,i_y-2:i_y+3,:] 
                         - concat_iterat(tensor_right_padding[:,i_x,i_y,:]))
            cross_y_right.append(tensor_right_padding[:,i_x-2:i_x+3,i_y-2:i_y+3,:] 
                         - concat_iterat(tensor_left_padding[:,i_x,i_y,:]))
        cross_x_left.append(K.concatenate(cross_y_left, axis=2))
        cross_x_right.append(K.concatenate(cross_y_right, axis=2))
        cross_y_left = []
        cross_y_right = []
    cross_out_left = K.concatenate(cross_x_left,axis=1)
    cross_out_right = K.concatenate(cross_x_right,axis=1)
    cross_out = K.concatenate([cross_out_left, cross_out_right], axis=3)
    return K.abs(cross_out)

def cross_input_single(X):
    tensor_left = X[0]
    tensor_right = X[1]
    x_length = K.int_shape(tensor_left)[1]
    y_length = K.int_shape(tensor_left)[2]
    cross_y = []
    cross_x = []
    tensor_left_padding = K.spatial_2d_padding(tensor_left,padding=(2,2))
    tensor_right_padding = K.spatial_2d_padding(tensor_right,padding=(2,2))
    for i_x in range(2, x_length + 2):
        for i_y in range(2, y_length + 2):
            cross_y.append(tensor_left_padding[:,i_x-2:i_x+3,i_y-2:i_y+3,:] 
                         - concat_iterat(tensor_right_padding[:,i_x,i_y,:]))
        cross_x.append(K.concatenate(cross_y,axis=2))
        cross_y = []
    cross_out = K.concatenate(cross_x,axis=1)
    return K.abs(cross_out)

def cross_input_shape_both(input_shapes):
    input_shape = input_shapes[0]
    return (input_shape[0],input_shape[1] * 5,input_shape[2] * 5,input_shape[3]*2)

def cross_input_shape_single(input_shapes):
    input_shape = input_shapes[0]
    return (input_shape[0],input_shape[1] * 5,input_shape[2] * 5,input_shape[3])    

'''
data preparing
-------------------------------------------------------------------------------
'''
print 'preparing training data......'
Data_Generator = pre_image.ImageDataGenerator(
    width_shift_range=0.05,
    height_shift_range=0.05)
    
X_pos_train,Y_pos_train = load_positive_data()
print 'already loaded positive data.'
print 'positive data number:',len(Y_pos_train)

Data_Generator.fit(X_pos_train[0],augment=True,rounds=5,seed=1217)
Data_Generator.fit(X_pos_train[1],augment=True,rounds=5,seed=1217)
Y_pos_train = np.repeat(Y_pos_train,5,axis=0)
print 'positive data augmentation done.'
print 'positive X data number after augmentation:',len(X_pos_train[0])
print 'positive Y data number after augmentation:',len(Y_pos_train)

X_neg_train,Y_neg_train = load_negative_data(len(X_pos_train[0]))
print 'already loaded negative data.'
print 'negative data number:',len(Y_neg_train)

X_train = []
X_train.append(np.concatenate([X_pos_train[0],X_neg_train[0]],axis=0))
X_train.append(np.concatenate([X_pos_train[1],X_neg_train[1]],axis=0))
Y_train = np.concatenate([Y_pos_train,Y_neg_train],axis=0)
Y_train = np_utils.to_categorical(Y_train, 2)
print 'data preprocessing done.'
print 'X train data number:',len(X_train[0])
print 'Y train data number:',len(Y_train)
'''
-------------------------------------------------------------------------------
'''



print 'now begin to compile the model.'
'''
model definition and compile
-------------------------------------------------------------------------------
'''
a1 = Input(shape=(160,60,3))
b1 = Input(shape=(160,60,3))
share = Convolution2D(20,5,5,dim_ordering='tf', W_regularizer=l2(l=0.0005))
a2 = share(a1)
b2 = share(b1)
a3 = Activation('relu')(a2)
b3 = Activation('relu')(b2)
a4 = MaxPooling2D(dim_ordering='tf')(a3)
b4 = MaxPooling2D(dim_ordering='tf')(b3)
share2 = Convolution2D(25,5,5,dim_ordering='tf', W_regularizer=l2(l=0.0005))
a5 = share2(a4)
b5 = share2(b4)
a6 = Activation('relu')(a5)
b6 = Activation('relu')(b5)
a7 = MaxPooling2D(dim_ordering='tf')(a6)
b7 = MaxPooling2D(dim_ordering='tf')(b6)
a8 = merge([a7,b7],mode=cross_input_single,output_shape=cross_input_shape_single)
b8 = merge([b7,a7],mode=cross_input_single,output_shape=cross_input_shape_single)
a9 = Convolution2D(25,5,5, subsample=(5,5), dim_ordering='tf',activation='relu', W_regularizer=l2(l=0.0005))(a8)
b9 = Convolution2D(25,5,5, subsample=(5,5), dim_ordering='tf',activation='relu', W_regularizer=l2(l=0.0005))(b8)
a10 = Convolution2D(25,3,3, subsample=(1,1), dim_ordering='tf',activation='relu', W_regularizer=l2(l=0.0005))(a9)
b10 = Convolution2D(25,3,3, subsample=(1,1), dim_ordering='tf',activation='relu', W_regularizer=l2(l=0.0005))(b9)
a11 = MaxPooling2D((2,2),dim_ordering='tf')(a10)
b11 = MaxPooling2D((2,2),dim_ordering='tf')(b10)
c1 = merge([a11, b11], mode='concat', concat_axis=-1)
c2 = Flatten()(c1)
c3 = Dense(500,activation='relu', W_regularizer=l2(l=0.0005))(c2)
c4 = Dense(2,activation='softmax', W_regularizer=l2(l=0.0005))(c3)

model = Model(input=[a1,b1],output=c4)
model.summary()

sgd = SGD_new(lr=0.01, momentum=0.9)

model.compile(optimizer=sgd,
loss='categorical_crossentropy',
metrics=['accuracy'])
print 'model compile done.'
'''
-------------------------------------------------------------------------------
'''

print 'begin to train model.'
model.fit_generator(NumpyArrayIterator_for_multiinput(X_train, Y_train, batch_size=150),
                    samples_per_epoch=len(X_train), nb_epoch=20)
