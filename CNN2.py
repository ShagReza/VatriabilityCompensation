
from keras.layers import Activation as activation
from keras.layers.core import Dense, Flatten , Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
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

path='D:\Shapar\ShaghayeghUni\AfterPropozal\Step1-EventLandmark\Programs\MyPrograms\EventExtraction\Keras'


#import data-------------------------------------------------------------------  
#train:
data_dict = mat73.loadmat(path+'\CB_total_train.mat')
x_train=np.transpose(data_dict['CB_total_train'])
LBL_total_train = scipy.io.loadmat(path+'\LBL_total_train.mat')
y_train=LBL_total_train['LBL_total_train']
#val:
CB_total_val= scipy.io.loadmat(path+'\CB_total_val.mat')
x_val=CB_total_val['CB_total_val']
LBL_total_val = scipy.io.loadmat(path+'\LBL_total_val.mat')
y_val=LBL_total_val['LBL_total_val']
#------------------------------------------------------------------------------
L=len(x_train)
x_train=x_train.reshape(L,15,18,1).swapaxes(1,2)

L=len(x_val)
x_val=x_val.reshape(L,15,18,1).swapaxes(1,2)
#------------------------------------------------------------------------------




#------------------------------------------------------------------------------
# CNN model:
L1=18;
L2=15 #context=7
input_tensor = Input(shape=(L1,L2,1))
x=input_tensor
cnns = (16,32,64,64)
kernel = (2,2,2,2)
pad = 'same'  ## 'same' or 'valid'
pool_size = (2,2)
strides = (2,2)


for i in range(len(cnns)-1):
    x = Convolution2D(cnns[i], kernel[i], padding=pad, kernel_initializer = 'he_normal', activation="relu")(x)
    x=BatchNormalization()(x)
    x=MaxPooling2D(pool_size, strides)(x)
    


#x=GlobalAveragePooling2D()(x)   # or faltten since seems not sufficient  
#x=Flatten()(x)   #7600 is too high
#x=AveragePooling2D()(x) 
#x=AveragePooling2D()(x) 
x=Flatten()(x)
x = Dense(1000 , activation="relu")(x)
x = Dense(1000 , activation="relu")(x)
x = Dense(1000 , activation="relu")(x)
x = Dense(103, activation="linear")(x)
         
output = x
model=Model(input_tensor, output)
model.summary()
#------------------------------------------------------------------------------







#------------------------------------------------------------------------------
# train parameters:
opt1 = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=opt1, loss='mean_squared_error', metrics= ['mean_squared_error'])
reduce_LR = callbacks.ReduceLROnPlateau(monitor='mean_squared_error', factor=0.1, patience=5, verbose=1, mode='max', min_delta=0.0001, cooldown=0, min_lr=0)

tensorboard1 = TensorBoard(path+'/log_dir')
logger = CSVLogger(path+'/training.log')
folder_model="./model"
path_model = os.path.join(folder_model, 'model_epoch_{epoch:02d}.hdf5')

model_checkpoint = callbacks.ModelCheckpoint(
    						filepath= path_model,
    						monitor="mean_squared_error",
    						mode="max",
    						verbose=0,
    						save_best_only=False,
                            save_weights_only=False)    

Early_stop=callbacks.EarlyStopping(monitor='mean_squared_error',
                                   min_delta=0.0001,
                                   patience=10,
                                   verbose=0,
                                   mode='auto',
                                   baseline=None)
#------------------------------------------------------------------------------




#------------------------------------------------------------------------------
history = model.fit(x_train, y_train, 
                    batch_size=128, 
                    epochs=40,
                    validation_data=(x_val, y_val),
                    shuffle=True,
                    verbose=1,
                    callbacks= [
                        model_checkpoint
                        , tensorboard1
                        , logger
                        , reduce_LR
                        , Early_stop
                        ]
                    )
model.save('LastModel.h5')

#------------------------------------------------------------------------------
PathTest=path+'\TestMat'
PathOut=path+'\TestOut'
import glob
D=glob.glob(PathTest+"/*.mat")
NumTest=len(D)
for i in range(NumTest):
    CB= scipy.io.loadmat(D[i])
    CB_value=CB['CB_context']

    L=len(CB_value)
    CB_value=CB_value.reshape(L,15,18,1).swapaxes(1,2)

    y_predict = model.predict(CB_value)
    
    
    
    D2=D[i].replace(PathTest, PathOut)
    scipy.io.savemat(D2, {'lbl': y_predict})
#------------------------------------------------------------------------------

  
    
    model = load_model('D:\Shapar\ShaghayeghUni\AfterPropozal\Step1-EventLandmark\Programs\MyPrograms\EventExtraction\Keras\model\model_epoch_22.hdf5')
