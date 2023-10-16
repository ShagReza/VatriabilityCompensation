# -*- coding: utf-8 -*-
"""
Embeding Forming With Best Landmarks
"""

#------------------------------------------------------------------------------
# Steps:
# 1: load all data and pretrian a forward model
# 2: change soft labels to hard labels 
# 3: find net output and extract best landmraks
# 4: train embeding on just landmark data with ئسث from best landmarks
# 5: train froward net again
# 6: loop 4 and 5 until convergence

# best landmark variations:
# 1: fixed with pretrain model 
# 2: change with each iteration
# 3: have a relationship with each class not the best recognized one 
#    for example mean of truely recognized ones (like diarization)
#------------------------------------------------------------------------------


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
#train:
path='D:\Shapar\ShaghayeghUni\AfterPropozal\MyPrograms\EventExtraction\Keras'
data_dict = mat73.loadmat(path+'\CB_total_train.mat')
x_train=np.transpose(data_dict['CB_total_train'])
LBL_total_train = scipy.io.loadmat(path+'\LBL_total_train.mat')
y_train=LBL_total_train['LBL_total_train']
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Dense model:
L=len(x_train[0])
input_tensor = Input(shape=(L,))
x=input_tensor
x = Dense(1000, activation="relu",name='v1')(x)
x = Dense(1000, activation="relu",name='v2')(x)
x = Dense(1000, activation="relu",name='v3')(x)
x = Dense(1000, activation="relu",name='v4')(x)
x = Dense(1000, activation="relu",name='v5')(x)
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
history = model1.fit(x_train, y_train, batch_size=128,  epochs=20, shuffle=True,verbose=1,
                    callbacks= [model_checkpoint, tensorboard1 , logger , reduce_LR, Early_stop] )
model1.save('PreModel.h5')
#------------------------------------------------------------------------------


'''
#------------------------------------------------------------------------------
LabelI = np.zeros(shape=(1,103))
LandmarkI5 = np.zeros(shape=(1,270))
Labels5=np.zeros(shape=(1,1))
for i in range(5):
    print(i)
    LabelI[0:102]=0
    LabelI[0,i]=1
    LabelI_index=np.where((y_train == LabelI).all(axis=1))[0]
    LandmrakI=x_train[LabelI_index]
    LandmarkI5=np.concatenate((LandmarkI5,LandmrakI), axis=0)
    Labels=np.zeros(shape=(1,np.shape(LabelI_index)[0]))+i
    Labels5=np.concatenate((Labels5,Labels), axis=1)
LandmarkI5=np.delete(LandmarkI5,0,0)
Labels5=np.delete(Labels5,0,1)
Labels5=Labels5[0]


from sklearn.decomposition import PCA
from TsneSpk import TsneSpk
pca = PCA(n_components=50)
pca.fit(LandmarkI5) 
LandmrakI_pca=pca.transform(LandmarkI5)
NumOutput=5
TsneSpk(LandmrakI_pca,Labels5,NumOutput)
# v4 or v5?????
intermediate_layer_model = Model(inputs=model1.input,outputs=model1.get_layer('v4').output)
intermediate_output = intermediate_layer_model.predict([LandmarkI5])
pca = PCA(n_components=50)
pca.fit(intermediate_output) 
LandmrakI_pca=pca.transform(intermediate_output)
NumOutput=5
TsneSpk(LandmrakI_pca,Labels5,NumOutput)
#------------------------------------------------------------------------------
'''




#------------------------------------------------------------------------------
# Fix y_train
'''
for i in range(np.shape(x_train)[0]):
    print(i)
    if max(y_train[i,0:101]>0):
        y_train[i,102]=0
np.save('y_train.py',y_train)        
'''  
y_train=np.load('y_train.npy')
#------------------------------------------------------------------------------

        
#------------------------------------------------------------------------------    
# Find best landmarks
from keras.models import load_model
model1 = load_model('PreModel.h5')

FinalLandmarkCode = scipy.io.loadmat(path+'\FinalLandmarkCode.mat')
LandmarkCode=FinalLandmarkCode['FinalLandmarkCode']
x_out=model1.predict(x_train)
SelectedLandmarks=np.zeros((1,270))
OutLandmarks=np.zeros((1,103))
BestLandmraks=np.zeros((1,270))
for i in range(np.shape(LandmarkCode)[0]):
    print(i)
    LabelI_index=np.where((y_train == LandmarkCode[i,:]).all(axis=1))[0]
    x_out_i=x_out[LabelI_index]
    mse_i=((x_out_i - LandmarkCode[i,:])**2).mean(axis=1)
    if (np.shape(mse_i)[0]>0):
        #----------------------
        # One best
        Argmin_mse=np.argmin(mse_i)
        ArgBest=LabelI_index[Argmin_mse]
        Best=x_train[ArgBest,:]
        # mean best
        # !!!!!!!!!
        #----------------------
        SelectedLandmarks=np.append(SelectedLandmarks, x_train[LabelI_index], axis=0)
        OutLandmarks=np.append(OutLandmarks, y_train[LabelI_index], axis=0)
        repeats_array = np.tile(Best, (np.shape(x_out_i)[0], 1))
        BestLandmraks=np.append(BestLandmraks, repeats_array, axis=0)
SelectedLandmarks=np.delete(SelectedLandmarks, 1, 0)
BestLandmraks=np.delete(BestLandmraks, 1, 0)
OutLandmarks=np.delete(OutLandmarks, 1, 0)
# Find embeding layer's output
intermediate_layer_model = Model(inputs=model1.input,outputs=model1.get_layer('v4').output)
EmbedingOfBestLandmarks = intermediate_layer_model.predict([BestLandmraks])
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Main  model with two outpus
L=len(x_train[0])
input_tensor = Input(shape=(L,))
x=input_tensor
x = Dense(1000, activation="relu",name='v1')(x)
x = Dense(1000, activation="relu",name='v2')(x)
x = Dense(1000, activation="relu",name='v3')(x)
x = Dense(1000, activation="relu",name='v4')(x)
y = Dense(1000, activation="relu",name='v5')(x)
y = Dense(103, activation="linear",name='v6')(y)
output = [x,y]
model=Model(input_tensor, output)
model.summary()

model.compile(loss={'v6': 'mean_squared_error', 'v4': 'mean_squared_error'},
              loss_weights={'v6': 0.2, 'v4': 0.8},
              optimizer=opt1)
#---------
# Train Main model
for i in range(20):
    # train on Best landmarks
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
    model.layers[6].set_weights(WeightL6)
    history = model.fit(SelectedLandmarks,[EmbedingOfBestLandmarks,OutLandmarks], batch_size=128,  epochs=1, shuffle=True,verbose=1)
    # train on main data
    WeightL1 = model.layers[1].get_weights()
    model1.layers[1].set_weights(WeightL1)
    WeightL2 = model.layers[2].get_weights()
    model1.layers[2].set_weights(WeightL2)
    WeightL3 = model.layers[3].get_weights()
    model1.layers[3].set_weights(WeightL3)
    WeightL4 = model.layers[4].get_weights()
    model1.layers[4].set_weights(WeightL4)
    WeightL5 = model.layers[5].get_weights()
    model1.layers[5].set_weights(WeightL5)
    WeightL6 = model.layers[6].get_weights()
    model1.layers[6].set_weights(WeightL6)
    history = model1.fit(x_train, y_train, batch_size=128,  epochs=1, shuffle=True,verbose=1)
#------------------------------------------------------------------------------



 intermediate_output = intermediate_layer_model.predict([LandmarkI5])
pca = PCA(n_components=50)
pca.fit(intermediate_output) 
LandmrakI_pca=pca.transform(intermediate_output)
NumOutput=5
TsneSpk(LandmrakI_pca,Labels5,NumOutput)   


  


#------------------------------------------------------------------------------
# To select Best landmarks again

for k in range(20):
    #------------------------------------------------------------------------------    
    # Find best landmarks
    FinalLandmarkCode = scipy.io.loadmat(path+'\FinalLandmarkCode.mat')
    LandmarkCode=FinalLandmarkCode['FinalLandmarkCode']
    x_out=model1.predict(x_train)
    SelectedLandmarks=np.zeros((1,270))
    OutLandmarks=np.zeros((1,103))
    BestLandmraks=np.zeros((1,270))
    for i in range(np.shape(LandmarkCode)[0]):
        print(i)
        LabelI_index=np.where((y_train == LandmarkCode[i,:]).all(axis=1))[0]
        x_out_i=x_out[LabelI_index]
        mse_i=((x_out_i - LandmarkCode[i,:])**2).mean(axis=1)
        if (np.shape(mse_i)[0]>10):
            #----------------------
            # One best
            
            Argmin_mse=np.argmin(mse_i)
            ArgBest=LabelI_index[Argmin_mse]
            Best=x_train[ArgBest,:]
            
            # mean best
            '''
            Argsort_mse=np.argsort(mse_i)
            ArgBest=LabelI_index[Argsort_mse[1:9]]
            Best=np.mean(x_train[ArgBest,:],axis=0)
            '''
            #----------------------
            SelectedLandmarks=np.append(SelectedLandmarks, x_train[LabelI_index], axis=0)
            OutLandmarks=np.append(OutLandmarks, y_train[LabelI_index], axis=0)
            repeats_array = np.tile(Best, (np.shape(x_out_i)[0], 1))
            BestLandmraks=np.append(BestLandmraks, repeats_array, axis=0)
    SelectedLandmarks=np.delete(SelectedLandmarks, 1, 0)
    BestLandmraks=np.delete(BestLandmraks, 1, 0)
    OutLandmarks=np.delete(OutLandmarks, 1, 0)
    # Find embeding layer's output
    intermediate_layer_model = Model(inputs=model1.input,outputs=model1.get_layer('v4').output)
    EmbedingOfBestLandmarks = intermediate_layer_model.predict([BestLandmraks])
    #------------------------------------------------------------------------------
        # train on Best landmarks
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
    model.layers[6].set_weights(WeightL6)
    history = model.fit(SelectedLandmarks,[EmbedingOfBestLandmarks,OutLandmarks], batch_size=128,  epochs=1, shuffle=True,verbose=1)
    # train on main data
    WeightL1 = model.layers[1].get_weights()
    model1.layers[1].set_weights(WeightL1)
    WeightL2 = model.layers[2].get_weights()
    model1.layers[2].set_weights(WeightL2)
    WeightL3 = model.layers[3].get_weights()
    model1.layers[3].set_weights(WeightL3)
    WeightL4 = model.layers[4].get_weights()
    model1.layers[4].set_weights(WeightL4)
    WeightL5 = model.layers[5].get_weights()
    model1.layers[5].set_weights(WeightL5)
    WeightL6 = model.layers[6].get_weights()
    model1.layers[6].set_weights(WeightL6)
    history = model1.fit(x_train, y_train, batch_size=128,  epochs=1, shuffle=True,verbose=1)
#------------------------------------------------------------------------------
    


model1.save('ّForming-PreModel-2.h5')
model.save('Forming-Model-2.h5')

model1 = load_model('ّForming-PreModel-2.h5')
model = load_model('Forming-Model-2.h5')












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







#------------------------------------------------------------------------------
# To Use a triplet-like metric
GRLstrength=0.5
import keras.backend as K
L=len(x_train[0])
input_tensor = Input(shape=(L,))
x=input_tensor
x = Dense(1000, activation="relu",name='v1')(x)
x = Dense(1000, activation="relu",name='v2')(x)
x = Dense(1000, activation="relu",name='v3')(x)
x = Dense(1000, activation="relu",name='v4')(x)
flip_layer = GradientReversal(GRLstrength,name='G')
z= flip_layer(x)
y = Dense(1000, activation="relu",name='v5')(x)
y = Dense(103, activation="linear",name='v6')(y)
output = [x,z,y]
model=Model(input_tensor, output)
model.summary()

model.compile(loss={'v6': 'mean_squared_error', 'v4': 'mean_squared_error', 'G': 'mean_squared_error'},
              loss_weights={'v6': 0.8, 'v4': 0.1, 'G': 0.1},
              optimizer=opt1)

#-----------------------------------------------------
# Find best landmarks
FinalLandmarkCode = scipy.io.loadmat(path+'\FinalLandmarkCode.mat')
LandmarkCode=FinalLandmarkCode['FinalLandmarkCode']
x_out=model1.predict(x_train)
SelectedLandmarks=np.zeros((1,270))
OutLandmarks=np.zeros((1,103))
BestLandmraks=np.zeros((1,270))
Attractors=np.zeros((1,270))
for i in range(np.shape(LandmarkCode)[0]):
    print(i)
    LabelI_index=np.where((y_train == LandmarkCode[i,:]).all(axis=1))[0]
    x_out_i=x_out[LabelI_index]
    mse_i=((x_out_i - LandmarkCode[i,:])**2).mean(axis=1)
    if (np.shape(mse_i)[0]>10):
        #----------------------
        # One best            
        Argmin_mse=np.argmin(mse_i)
        ArgBest=LabelI_index[Argmin_mse]
        Best=x_train[ArgBest,:]
        #----------------------
        SelectedLandmarks=np.append(SelectedLandmarks, x_train[LabelI_index], axis=0)
        OutLandmarks=np.append(OutLandmarks, y_train[LabelI_index], axis=0)
        repeats_array = np.tile(Best, (np.shape(x_out_i)[0], 1))
        BestLandmraks=np.append(BestLandmraks, repeats_array, axis=0)
        #---
        repeats_array = np.tile(Best, (1, 1))
        Attractors=np.append(Attractors, repeats_array, axis=0)
        
SelectedLandmarks=np.delete(SelectedLandmarks, 1, 0)
BestLandmraks=np.delete(BestLandmraks, 1, 0)
OutLandmarks=np.delete(OutLandmarks, 1, 0)
Attractors=np.delete(Attractors, 1, 0)   
# Find embeding layer's output
intermediate_layer_model = Model(inputs=model1.input,outputs=model1.get_layer('v4').output)
EmbedingOfBestLandmarks = intermediate_layer_model.predict([BestLandmraks])
#------------------------------------------------------------------------------
#Find negative outputs:
NegativeAttractors=np.zeros((1,270))
for i in range(np.shape(SelectedLandmarks)[0]):
    print(i)
    mse_i=((BestLandmraks[i,:] - Attractors)**2).mean(axis=1)
    IndexSort = np.argsort(mse_i)
    repeats_array = np.tile(Attractors[IndexSort[1],:], (1, 1))
    NegativeAttractors=np.append(NegativeAttractors, repeats_array, axis=0)
NegativeAttractors=np.delete(NegativeAttractors, 1, 0)
intermediate_layer_model = Model(inputs=model1.input,outputs=model1.get_layer('v4').output)
EmbedingOfNegativeAttractors = intermediate_layer_model.predict([NegativeAttractors])
np.save('EmbedingOfNegativeAttractors.py',EmbedingOfNegativeAttractors),knm  
#------------------------------------------------------------------------------------
    
for k in range(20):  
    # train on Best landmarks
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
    history = model.fit(SelectedLandmarks,[EmbedingOfBestLandmarks, EmbedingOfNegativeAttractors,OutLandmarks], batch_size=128,  epochs=1, shuffle=True,verbose=1)
    # train on main data
    WeightL1 = model.layers[1].get_weights()
    model1.layers[1].set_weights(WeightL1)
    WeightL2 = model.layers[2].get_weights()
    model1.layers[2].set_weights(WeightL2)
    WeightL3 = model.layers[3].get_weights()
    model1.layers[3].set_weights(WeightL3)
    WeightL4 = model.layers[4].get_weights()
    model1.layers[4].set_weights(WeightL4)
    WeightL5 = model.layers[5].get_weights()
    model1.layers[5].set_weights(WeightL5)
    WeightL6 = model.layers[7].get_weights()
    model1.layers[6].set_weights(WeightL6)
    history = model1.fit(x_train, y_train, batch_size=128,  epochs=1, shuffle=True,verbose=1)
#------------------------------------------------------------------------------
    

for k in range(20):  
    history = model.fit(SelectedLandmarks,[EmbedingOfBestLandmarks, EmbedingOfNegativeAttractors,OutLandmarks], batch_size=128,  epochs=1, shuffle=True,verbose=1)
    # train on main data
    
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer('v4').output)
intermediate_output = intermediate_layer_model.predict([LandmarkI5])
pca = PCA(n_components=50)
pca.fit(intermediate_output) 
LandmrakI_pca=pca.transform(intermediate_output)
NumOutput=5
TsneSpk(LandmrakI_pca,Labels5,NumOutput)







#------------------------------------------------------------------------------
# To Use only between class metric
GRLstrength=0.5
import keras.backend as K
L=len(x_train[0])
input_tensor = Input(shape=(L,))
x=input_tensor
x = Dense(1000, activation="relu",name='v1')(x)
x = Dense(1000, activation="relu",name='v2')(x)
x = Dense(1000, activation="relu",name='v3')(x)
x = Dense(1000, activation="relu",name='v4')(x)
flip_layer = GradientReversal(GRLstrength,name='G')
z= flip_layer(x)
y = Dense(1000, activation="relu",name='v5')(x)
y = Dense(103, activation="linear",name='v6')(y)
output = [z,y]
model=Model(input_tensor, output)
model.summary()

model.compile(loss={'v6': 'mean_squared_error',  'G': 'mean_squared_error'},
              loss_weights={'v6': 0.8, 'G': 0.2},
              optimizer=opt1)

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

for k in range(20):  
    history = model.fit(SelectedLandmarks,[EmbedingOfNegativeAttractors,OutLandmarks], batch_size=128,  epochs=1, shuffle=True,verbose=1)
    # train on main data
    
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer('v4').output)
intermediate_output = intermediate_layer_model.predict([LandmarkI5])
pca = PCA(n_components=50)
pca.fit(intermediate_output) 
LandmrakI_pca=pca.transform(intermediate_output)
NumOutput=5
TsneSpk(LandmrakI_pca,Labels5,NumOutput)
