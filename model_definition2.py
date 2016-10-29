# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 14:39:54 2016

@author: lenovo
"""

from keras.models import Model
from keras.layers import Input,Dense,Convolution2D,Activation,MaxPooling2D,Flatten,merge
from keras.regularizers import l2
from keras import backend as K

print 'now begin to define the model.'


def cross_input(X):
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
                         - tensor_right_padding[:,i_x-2:i_x+3,i_y-2:i_y+3,:])
        cross_x.append(K.concatenate(cross_y, axis=2))
        cross_y = []
    cross_out = K.concatenate(cross_x,axis=1)
    return K.abs(cross_out)

def cross_input_shape(input_shapes):
    input_shape = input_shapes[0]
    return (input_shape[0],input_shape[1] * 5,input_shape[2] * 5,input_shape[3])



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
a8 = merge([a7,b7],mode=cross_input,output_shape=cross_input_shape)
b8 = merge([b7,a7],mode=cross_input,output_shape=cross_input_shape)
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

print 'model definition done.'