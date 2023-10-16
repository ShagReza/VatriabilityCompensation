# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 00:29:14 2020

@author: user
"""

#Data partitioning

import scipy.io
from keras.utils.np_utils import to_categorical
import numpy as np
import mat73
#------------------------------------------------------------------------------

path='D:\Shapar\ShaghayeghUni\AfterPropozal\MyPrograms\EventExtraction\Keras'
CB_LandamarksTrain_Just11= scipy.io.loadmat(path+'\CB_LandamarksTrain.mat')
x_train=CB_LandamarksTrain_Just11['CB_LandamarksTrain']

CB_BestLandmarksTrain_Just11= scipy.io.loadmat(path+'\CB_BestLandmarksTrain.mat')
y_train=CB_BestLandmarksTrain_Just11['CB_BestLandmarksTrain']


#------------------------------------------------------------------------------
#randomize data  befor partitioning
r=np.random.permutation(np.shape(x_train)[0])
x_train=x_train[r]
y_train=y_train[r]

# Partition data
NumParts=5
LenPart=np.shape(x_train)[0]//NumParts
for i in range(NumParts):
    NamePartX='TrainLandmarkPart'+ str(i)
    NamePartY='BestLandmarkPart'+ str(i)
    np.save(NamePartX,x_train[0:LenPart,:])
    x_train=np.delete(x_train,range(LenPart), axis=0)
    np.save(NamePartY,y_train[0:LenPart,:])
    y_train=np.delete(y_train,range(LenPart), axis=0)
np.save('LenPart.npy',LenPart)
np.save('NumParts.npy',NumParts)
#------------------------------------------------------------------------------

    
