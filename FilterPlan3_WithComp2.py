
#------------------------------------------------------------------------------
from keras.layers import Activation as activation
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
#------------------------------------------------------------------------------


#import data-------------------------------------------------------------------  

path='D:\Shapar\ShaghayeghUni\AfterPropozal\Step1-EventLandmark\Programs\MyPrograms\EventExtraction\Keras'

#val:
CB_total_val= scipy.io.loadmat(path+'\CB_total_val.mat')
x_val=CB_total_val['CB_total_val']
LBL_total_val = scipy.io.loadmat(path+'\LBL_total_val.mat')
y_val=LBL_total_val['LBL_total_val']
#------------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# after running  (LandmarkNumber_Train7Test.m):
# load matlab output:
data_dict = mat73.loadmat(path+'\CBtrain_In_Comp2.mat')
x_train=np.transpose(data_dict['CBtrain_In_Comp2'])


data_dict = mat73.loadmat(path+'\CBtrain_Out_Comp2.mat')
x_train_out=np.transpose(data_dict['CBtrain_Out_Comp2'])

LBL_out_Comp2= scipy.io.loadmat(path+'\LBL_out_Comp2.mat')
train_lbl=LBL_out_Comp2['LBL_out_Comp2']
#-----------------------------------------------------------------------------




#------------------------------------------------------------------------------
L=len(x_train[0])
input_tensor = Input(shape=(L,))
x=input_tensor
x = Dense(500, activation="relu")(x)
x = Dense(400, activation="relu")(x)
x2= Dense(300, activation="relu")(x)
x2= Dense(270, activation="linear", name='out1')(x2)
outDae=x2;
x = Dense(300, activation="relu")(x)
x = Dense(300, activation="relu")(x)
x = Dense(500, activation="relu")(x)
x = Dense(300, activation="relu")(x)
x = Dense(103, activation="linear",name='out2')(x)
output = x
model=Model(input_tensor, [outDae, output])
model.summary()
#------------------------------------------------------------------------------





#------------------------------------------------------------------------------
L=len(x_train[0])
input_tensor = Input(shape=(L,))
x=input_tensor
x = Dense(700, activation="relu")(x)
x = Dense(700, activation="relu")(x)
x2= Dense(700, activation="relu")(x)
x2= Dense(270, activation="linear", name='out1')(x2)
outDae=x2;
#x = Dense(1000, activation="relu")(x)
#x = Dense(1000, activation="relu")(x)
x = Dense(1000, activation="relu")(x)
x = Dense(1000, activation="relu")(x)
x = Dense(1000, activation="relu")(x)
x = Dense(103, activation="linear",name='out2')(x)
output = x
model=Model(input_tensor, [outDae, output])
model.summary()
#-------------------





#------------------------------------------------------------------------------
"""
L=len(x_train[0])
input_tensor = Input(shape=(L,))
x=input_tensor
x = Dense(700, activation="relu")(x)
x = Dense(700, activation="relu")(x)
x= Dense(700, activation="relu")(x)
x= Dense(270, activation="linear", name='out1')(x)
outDae=x;
x = Dense(1000, activation="relu")(x)
x = Dense(1000, activation="relu")(x)
x = Dense(1000, activation="relu")(x)  
x = Dense(1000, activation="relu")(x)
x = Dense(1000, activation="relu")(x)
x = Dense(103, activation="linear",name='out2')(x)
output = x
model=Model(input_tensor, [outDae, output])
model.summary()
"""





#------------------------------------------------------------------------------
opt1 = optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss={'out1': 'mean_squared_error', 'out2': 'mean_squared_error'},
              loss_weights={'out1': 0.1, 'out2': 0.9},
              optimizer=opt1)

reduce_LR = callbacks.ReduceLROnPlateau(monitor='mean_squared_error', factor=0.1, patience=5, verbose=1, mode='max', min_delta=0.0001, cooldown=0, min_lr=0)
tensorboard1 = TensorBoard(path+'/log_dir')
logger = CSVLogger(path+'/training.log')
folder_model="./model"
path_model = os.path.join(folder_model, 'model_epoch_{epoch:02d}.hdf5')
model_checkpoint = callbacks.ModelCheckpoint(filepath= path_model,monitor="mean_squared_error", mode="max", verbose=0, save_best_only=False, save_weights_only=False)    
Early_stop=callbacks.EarlyStopping(monitor='mean_squared_error', min_delta=0.0001, patience=10, verbose=0, mode='auto', baseline=None)


# train model  
history = model.fit(x_train, [x_train_out,train_lbl], batch_size=128, epochs=500, shuffle=True, verbose=1,
                    validation_data=(x_val, [x_val,y_val]),
                    callbacks= [model_checkpoint, tensorboard1, logger, reduce_LR, Early_stop]
                    )

"""
history = model.fit(x_train, [x_train_out,train_lbl], batch_size=128, epochs=500, shuffle=True, verbose=1,
                    validation_data=(x_train, [x_train_out,train_lbl]),
                    callbacks= [model_checkpoint, tensorboard1, logger, reduce_LR, Early_stop]
                    )
"""
model.save('ModelManifoldLearningnew.h5')  
#------------------------------------------------------------------------------


from keras.models import load_model
model = load_model('D:\Shapar\ShaghayeghUni\AfterPropozal\Step1-EventLandmark\Programs\MyPrograms\EventExtraction\Keras\model\model_epoch_09.hdf5')

#------------------------------------------------------------------------------  
PathTest=path+'\TestMat'
PathOut=path+'\TestOut'
import glob
D=glob.glob(PathTest+"/*.mat")
NumTest=len(D)
for i in range(NumTest):
    CB= scipy.io.loadmat(D[i])
    CB_value=CB['CB_context']
    y_predict = model.predict(CB_value)[1]
    D2=D[i].replace(PathTest, PathOut)
    scipy.io.savemat(D2, {'lbl': y_predict})
#------------------------------------------------------------------------------
