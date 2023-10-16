# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 14:00:48 2020

@author: user
"""
#------------------------------------------------------------------------------
from keras.models import load_model
from keras.layers import Activation as activation
from keras.layers import Concatenate
from keras.layers.core import Dense, Flatten , Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras import layers
from keras import Input
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.optimizers import *
from keras import callbacks
from keras.callbacks import TensorBoard, CSVLogger
import os
import scipy.io
import h5py
import mat73
from keras import optimizers
from keras.layers import Lambda
import keras.backend as K
from tensorflow import keras
#------------------------------------------------------------------------------

path='D:\Shapar\ShaghayeghUni\AfterPropozal\MyPrograms\EventExtraction\Keras'
CB_LandamarksTrain= scipy.io.loadmat(path+'\CB_LandamarksTrain.mat')
x_train=CB_LandamarksTrain['CB_LandamarksTrain']
CB_BestLandmarksTrain= scipy.io.loadmat(path+'\CB_BestLandmarksTrain.mat')
y_train=CB_BestLandmarksTrain['CB_BestLandmarksTrain']

'''
data_dict = mat73.loadmat(path+'\CB_LandamarksTrain.mat')
x_train=np.transpose(data_dict['CB_LandamarksTrain'])

data_dict = mat73.loadmat(path+'\CB_BestLandmarksTrain.mat')
y_train=np.transpose(data_dict['CB_BestLandmarksTrain'])
'''


