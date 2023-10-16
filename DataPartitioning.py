#Data partitioning

import scipy.io
from keras.utils.np_utils import to_categorical
import numpy as np
import mat73
#------------------------------------------------------------------------------
#input
'''
path='D:\Shapar\ShaghayeghUni\AfterPropozal\MyPrograms\EventExtraction\Keras\lhcbDelta2'
# load matlab output (Just 11 landmarks):
CB_LandamarksTrain_Just11= scipy.io.loadmat(path+'\CB_LandamarksTrain_Just11.mat')
x_train=CB_LandamarksTrain_Just11['CB_LandamarksTrain_Just11']
LandmarksLabels_Just11 = scipy.io.loadmat(path+'\LandmarksLabels_Just11.mat')
labels=LandmarksLabels_Just11['LandmarksLabels_Just11'][0]
y_train= to_categorical(labels)
'''
path='D:\Shapar\ShaghayeghUni\AfterPropozal\MyPrograms\EventExtraction\Keras'
data_dict = mat73.loadmat(path+'\CB_total_train.mat')
x_train=np.transpose(data_dict['CB_total_train'])
LBL_total_train = scipy.io.loadmat(path+'\LBL_total_train.mat')
y_train=LBL_total_train['LBL_total_train']


#------------------------------------------------------------------------------
#randomize data  befor partitioning
r=np.random.permutation(np.shape(x_train)[0])
x_train=x_train[r]
y_train=y_train[r]

# Partition data
NumParts=5
LenPart=np.shape(x_train)[0]//NumParts
for i in range(NumParts):
    NamePartX='TrainPart'+ str(i)
    NamePartY='LabelPart'+ str(i)
    np.save(NamePartX,x_train[0:LenPart,:])
    x_train=np.delete(x_train,range(LenPart), axis=0)
    np.save(NamePartY,y_train[0:LenPart,:])
    y_train=np.delete(y_train,range(LenPart), axis=0)
np.save('LenPart.npy',LenPart)
np.save('NumParts.npy',NumParts)
#------------------------------------------------------------------------------
'''
# load partitioned data
LenPart=np.load('LenPart.npy')
NumParts=np.load('NumParts.npy')
for i in range(NumParts):
    NamePartX='TrainPart'+ str(i) + '.npy'
    NamePartY='LabelPart'+ str(i) + '.npy'
    x_train=np.load(NamePartX)
    y_train=np.load(NamePartY)
    '''
#------------------------------------------------------------------------------
    
