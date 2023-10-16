# -*- coding: utf-8 -*-
"""
Adversarial compensation: Gender
"""

#------------------------------------------------------------------------------
import tensorflow as tf
from keras.engine import Layer
import keras.backend as K


def reverse_gradient(X, hp_lambda):
    '''Flips the sign of the incoming gradient during training.'''
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]

    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y

class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = hp_lambda

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'hp_lambda': self.hp_lambda}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
#-------------------------------------------------------------------------------











#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use id from $ nvidia-smi

#To force using CPU
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""



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

path='D:\Shapar\ShaghayeghUni\AfterPropozal\MyPrograms\EventExtraction\Keras'


#import data-------------------------------------------------------------------  
#train:
data_dict = mat73.loadmat(path+'\CB_total_train.mat')
x_train=np.transpose(data_dict['CB_total_train'])
LBL_total_train = scipy.io.loadmat(path+'\LBL_total_train.mat')
y_train=LBL_total_train['LBL_total_train']

Gender_train = scipy.io.loadmat(path+'\Gender_train.mat')
G_train=Gender_train['Gender_train']
G_train=to_categorical(G_train,2)[0]


SpeakerNum_train = scipy.io.loadmat(path+'\SpeakerNum_train.mat')
S_train=SpeakerNum_train['SpeakerNum_train']
S_train=to_categorical(S_train)[0]

#val:
'''
CB_total_val= scipy.io.loadmat(path+'\CB_total_val.mat')
x_val=CB_total_val['CB_total_val']
LBL_total_val = scipy.io.loadmat(path+'\LBL_total_val.mat')
y_val=LBL_total_val['LBL_total_val']
Gender_val = scipy.io.loadmat(path+'\Gender_val.mat')
G_val=Gender_val['Gender_val']
G_val=to_categorical(G_val,2)
'''
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Dense model:
L=len(x_train[0])
input_tensor = Input(shape=(L,))
x=input_tensor
x = Dense(500, activation="relu",name='v1')(x)
x = Dense(500, activation="relu",name='v2')(x)
x = Dense(500, activation="relu",name='v3')(x)
x = Dense(500, activation="relu",name='v4')(x)
x = Dense(500, activation="relu",name='v5')(x)
x = Dense(103, activation="linear",name='v6')(x)
output = x
model1=Model(input_tensor, output)
model1.summary()
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
opt1 = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model1.compile(optimizer=opt1, loss='mean_squared_error', metrics= ['mean_squared_error'])
reduce_LR = callbacks.ReduceLROnPlateau(monitor='mean_squared_error', factor=0.1, patience=5, verbose=1, mode='max', min_delta=0.0001, cooldown=0, min_lr=0)
tensorboard1 = TensorBoard(path+'/log_dir')
logger = CSVLogger(path+'/training.log')
folder_model="./model"
path_model = os.path.join(folder_model, 'model_epoch_{epoch:02d}.hdf5')
model_checkpoint = callbacks.ModelCheckpoint(filepath= path_model,monitor="mean_squared_error", mode="max", verbose=0, save_best_only=False, save_weights_only=False)    
Early_stop=callbacks.EarlyStopping(monitor='mean_squared_error', min_delta=0.0001, patience=10, verbose=0, mode='auto', baseline=None)
history = model1.fit(x_train, y_train, batch_size=128,  epochs=10, shuffle=True,verbose=1,
                    callbacks= [model_checkpoint, tensorboard1 , logger , reduce_LR, Early_stop] )
model1.save('PreModel_GnederCompensation.h5')
#------------------------------------------------------------------------------



from keras.models import load_model
model1 = load_model('PreModel_GnederCompensation.h5')




#------------------------------------------------------------------------------
# Dense model:
GRLstrength=0.1
L=len(x_train[0])
input_tensor = Input(shape=(L,))
x=input_tensor
x = Dense(500, activation="relu",name='w1')(x)
x = Dense(500, activation="relu",name='w2')(x)
x = Dense(500, activation="relu",name='w3')(x)
x = Dense(500, activation="relu",name='w4')(x)
flip_layer = GradientReversal(GRLstrength)
y= flip_layer(x)
y= Dense(units=400, activation='relu', name='S1')(y)
y= Dense(units=400, activation='relu', name='S2')(y)
y= Dense(units=304, activation='softmax', name='domain_classifier')(y)
x = Dense(500, activation="relu",name='w5')(x)
x = Dense(500, activation="relu",name='w55')(x)
z= Dense(103, activation="linear",name='w6')(x)
output = [z,y]
model=Model(input_tensor, output)
model.summary()
#------------------------------------------------------------------------------




#------------------------------------------------------------------------------
#use pretrain weights:
'''
WeightL1 = model1.layers[1].get_weights()
model.layers[1].set_weights(WeightL1)
WeightL2 = model1.layers[2].get_weights()
model.layers[2].set_weights(WeightL2)
WeightL3 = model1.layers[3].get_weights()
model.layers[3].set_weights(WeightL3)
WeightL4 = model1.layers[4].get_weights()
model.layers[4].set_weights(WeightL4)
WeightL5 = model1.layers[5].get_weights()
model.layers[5].set_weights(WeightL5)
WeightL6 = model1.layers[6].get_weights()
model.layers[7].set_weights(WeightL6)
'''

WeightL1 = model1.layers[1].get_weights()
model.layers[1].set_weights(WeightL1)
WeightL2 = model1.layers[2].get_weights()
model.layers[2].set_weights(WeightL2)
WeightL3 = model1.layers[3].get_weights()
model.layers[3].set_weights(WeightL3)
WeightL4 = model1.layers[4].get_weights()
model.layers[4].set_weights(WeightL4)
WeightL5 = model1.layers[5].get_weights()
model.layers[6].set_weights(WeightL5)
WeightL6 = model1.layers[6].get_weights()
model.layers[10].set_weights(WeightL6)

#------------------------------------------------------------------------------




#------------------------------------------------------------------------------  
# train parameters:
opt1 = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#model.compile(optimizer=opt1, loss='mean_squared_error', metrics= ['mean_squared_error'])
model.compile(loss={'w6': 'mean_squared_error', 'domain_classifier': 'categorical_crossentropy'},
              loss_weights={'w6': 0.8, 'domain_classifier': 0.2},
              optimizer=opt1)
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
history = model.fit(x_train, [y_train,S_train], 
                    batch_size=128, 
                    epochs=40,                    
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
model.save('model_SpeakerCompensation.h5')
#------------------------------------------------------------------------------






   


#------------------------------------------------------------------------------
        # Tsne for Genders
model1 = load_model('PreModel_GnederCompensation.h5')
model = load_model('model_GenderCompensation.h5')
from sklearn.decomposition import PCA
from TsneSpk import TsneSpk
SpeakerNum_train = scipy.io.loadmat(path+'\SpeakerNum_train.mat')
S_train0=SpeakerNum_train['SpeakerNum_train'][0]
# find landmark i of different genders 
LabelI=[]
LabelI=y_train[0]
LabelI[29]=0
LabelI[0]=1
LabelI_index=np.where((y_train == LabelI).all(axis=1))[0]
LandmrakI=x_train[LabelI_index]
SpeakerI=S_train0[LabelI_index] 

ii=np.where((SpeakerI == 5))[0][1]
LandmrakI10=LandmrakI[0:ii-1]
SpeakerI10=SpeakerI[0:ii-1]




# draw tsne
pca = PCA(n_components=50)
pca.fit(LandmrakI10) 
LandmrakI_pca=pca.transform(LandmrakI10)
NumOutput=max(SpeakerI10)
TsneSpk(LandmrakI_pca,SpeakerI10,NumOutput)
# find the embedings of model 1 (without adversarial)
# draw them
intermediate_layer_model = Model(inputs=model1.input,outputs=model1.get_layer('v4').output)
intermediate_output = intermediate_layer_model.predict([LandmrakI10])
pca = PCA(n_components=50)
pca.fit(intermediate_output) 
intermediate_output_pca=pca.transform(intermediate_output)
TsneSpk(intermediate_output_pca,SpeakerI10,NumOutput)
# find the embedings of model (with adversarial)
# draw them
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer('w4').output)
intermediate_output = intermediate_layer_model.predict([LandmrakI10])
pca = PCA(n_components=50)
pca.fit(intermediate_output) 
intermediate_output_pca=pca.transform(intermediate_output)
TsneSpk(intermediate_output_pca,SpeakerI10,NumOutput)
#------------------------------------------------------------------------------






'''
# for 5 speakers with all their landmarks
ii=np.where((S_train0 == 5))[0][1]
LandmrakI10=LandmrakI[0:ii-1]
SpeakerI10=SpeakerI[0:ii-1]

# draw tsne
pca = PCA(n_components=50)
pca.fit(LandmrakI10) 
LandmrakI_pca=pca.transform(LandmrakI10)
NumOutput=max(SpeakerI10)
TsneSpk(LandmrakI_pca,SpeakerI10,NumOutput)
# find the embedings of model 1 (without adversarial)
# draw them
intermediate_layer_model = Model(inputs=model1.input,outputs=model1.get_layer('v4').output)
intermediate_output = intermediate_layer_model.predict([LandmrakI10])
pca = PCA(n_components=50)
pca.fit(intermediate_output) 
intermediate_output_pca=pca.transform(intermediate_output)
TsneSpk(intermediate_output_pca,SpeakerI10,NumOutput)
# find the embedings of model (with adversarial)
# draw them
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer('w4').output)
intermediate_output = intermediate_layer_model.predict([LandmrakI10])
pca = PCA(n_components=50)
pca.fit(intermediate_output) 
intermediate_output_pca=pca.transform(intermediate_output)
TsneSpk(intermediate_output_pca,SpeakerI10,NumOutput)
'''



'''
!!!!!NOtTE: Its not a good idea to plot tsne with large number of classes and use it to visualize parts of it
    # for 5 speakers with all their landmarks
from TsneOnAllPLotOn5 import TsneSpk
LabelI=[]
LabelI=y_train[0]
LabelI[29]=0
LabelI[0]=1
LabelI_index=np.where((y_train == LabelI).all(axis=1))[0]
LandmrakI=x_train[LabelI_index]
SpeakerI=S_train0[LabelI_index] 

ii=np.where((SpeakerI == 5))[0][1]
LandmrakI10=LandmrakI[0:ii-1]
SpeakerI10=SpeakerI[0:ii-1]

# draw tsne
pca = PCA(n_components=50)
pca.fit(LandmrakI) 
LandmrakI_pca=pca.transform(LandmrakI)
NumOutput=max(SpeakerI10)
TsneSpk(LandmrakI_pca,SpeakerI10,NumOutput)
# find the embedings of model 1 (without adversarial)
# draw them
intermediate_layer_model = Model(inputs=model1.input,outputs=model1.get_layer('v4').output)
intermediate_output = intermediate_layer_model.predict([LandmrakI])
pca = PCA(n_components=50)
pca.fit(intermediate_output) 
intermediate_output_pca=pca.transform(intermediate_output)
TsneSpk(intermediate_output_pca,SpeakerI10,NumOutput)
# find the embedings of model (with adversarial)
# draw them
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer('w4').output)
intermediate_output = intermediate_layer_model.predict([LandmrakI])
pca = PCA(n_components=50)
pca.fit(intermediate_output) 
intermediate_output_pca=pca.transform(intermediate_output)
TsneSpk(intermediate_output_pca,SpeakerI10,NumOutput)
'''





























