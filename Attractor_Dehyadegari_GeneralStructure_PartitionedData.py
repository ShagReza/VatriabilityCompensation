# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:36:31 2020

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 23:45:26 2020

@author: Shaghayegh Reza
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
def ReLU(x):
    return x * (x > 0)
#------------------------------------------------------------------------------
    


#------------------------------------------------------------------------------
#input
LenPart=np.load('LenPart.npy')
NumParts=np.load('NumParts.npy')
i=0;
NamePartX='TrainPart'+ str(i) + '.npy'
NamePartY='LabelPart'+ str(i) + '.npy'
x_train=np.load(NamePartX)
y_train=np.load(NamePartY)
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
#points:will be diverged with relu function
# decrease lr?  خیلی در عدم واگرایی موثر بود
NumH1=500
NumH2=500
NumH3=250
NumH4=100
NumH5=11
NumH6=100
NumH7=100
landa=0.7
NumData=x_train.shape[0]
NumOut=y_train.shape[1]
#-------------------------------------------------
#inputs:
L=len(x_train[0])

input_1 = Input(shape=(L,))
input_y2 = Input(shape=(NumH2,))
input_y3 = Input(shape=(NumH3,))

for i in range(NumParts):
    y2before=0.1*np.random.rand(np.shape(x_train)[0],NumH2) 
    Name='y2beforePart'+ str(i)
    np.save(Name,y2before)
del(y2before)
for i in range(NumParts):
    y3before=0.1*np.random.rand(np.shape(x_train)[0],NumH3) 
    Name='y3beforePart'+ str(i)
    np.save(Name,y3before)
del(y3before)
    
'''
y1before=0.1*np.random.rand(np.shape(x_train)[0],NumH1) 
input_y1 = Input(shape=(NumH1,))
y4before=0.1*np.random.rand(np.shape(x_train)[0],NumH4) 
input_y4 = Input(shape=(NumH4,))
y5before=0.1*np.random.rand(np.shape(x_train)[0],NumH5) 
input_y5 = Input(shape=(NumH5,))
y6before=0.1*np.random.rand(np.shape(x_train)[0],NumH6) 
input_y6 = Input(shape=(NumH6,))
y7before=0.1*np.random.rand(np.shape(x_train)[0],NumH7) 
input_y7 = Input(shape=(NumH7,))
'''
#-------------------------------------------------
#layers:
def Merger(ip):
    landa=0.7
    a = ip[0]
    b=ip[1]
    return tf.keras.backend.tanh((1-landa)*a+ b)

#
v1=Dense(NumH1, activation="linear",name='v1')
v1a=Dense(NumH1, activation="tanh",name='v1a')
f1=Dense(NumH1, activation="linear",name='f1') 
L1=Lambda(Merger,name='L1')
#
v2=Dense(NumH2, activation="linear",name='v2')
v2a=Dense(NumH2, activation="tanh",name='v2a')
f2=Dense(NumH2, activation="linear",name='f2') 
L2=Lambda(Merger,name='L2')
#
v3=Dense(NumH3, activation="linear",name='v3')
v3a=Dense(NumH3, activation="tanh",name='v3a')
f3=Dense(NumH3, activation="linear",name='f3') 
L3=Lambda(Merger,name='L3')
#
v4=Dense(NumH4, activation="linear",name='v4')
v4a=Dense(NumH4, activation="tanh",name='v4a')
f4=Dense(NumH4, activation="linear",name='f4') 
L4=Lambda(Merger,name='L4')
#
v5=Dense(NumH5, activation="linear",name='v5')
v5a=Dense(NumH5, activation="tanh",name='v5a')
f5=Dense(NumH5, activation="linear",name='f5') 
L5=Lambda(Merger,name='L5')
#
v6=Dense(NumH6, activation="linear",name='v6')
v6a=Dense(NumH6, activation="tanh",name='v6a')
f6=Dense(NumH6, activation="linear",name='f6') 
L6=Lambda(Merger,name='L6')
#
v7=Dense(NumH7, activation="linear",name='v7')
v7a=Dense(NumH7, activation="tanh",name='v7a')
f7=Dense(NumH7, activation="linear",name='f7') 
L7=Lambda(Merger,name='L7')
#-------------------------------------------------
#net:
x=input_1
y1=v1a(x)
#
y2_forward=v2(y1)
y2_before=input_y2
y2_feedback=f2(y2_before)
y2=L2([y2_forward,y2_feedback])
#
y3_forward=v3(y2)
y3_before=input_y3
y3_feedback=f3(y3_before)
y3=L3([y3_forward,y3_feedback])
#
y4=v4a(y3)
y5=v5a(y4)
#
z=y5
#
model=Model([input_1,input_y2,input_y3], [y2_forward,y3_forward,z])
model.summary()
#-----------------------------------------------------------------------------
# train:
opt1 = optimizers.Adam(lr=0.00002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss={'v2': 'mean_squared_error', 'v3': 'mean_squared_error','v5a': 'mean_squared_error'},
              loss_weights={'v2': 0.2, 'v3': 0.2, 'v5a': 0.6},optimizer=opt1)
reduce_LR = callbacks.ReduceLROnPlateau(monitor='mean_squared_error', factor=0.1, patience=5, verbose=1, mode='max', min_delta=0.0001, cooldown=0, min_lr=0)
#
Feedback2 = Model(inputs=model.input,outputs=model.get_layer('f2').output)
Feedback3 = Model(inputs=model.input,outputs=model.get_layer('f3').output)
#-----------------------------------------------------------------------------  
for i in range(NumParts):
    print(i)
    NamePartX='TrainPart'+ str(i) + '.npy'
    NamePartY='LabelPart'+ str(i) + '.npy'
    Name_y2before='y2beforePart'+ str(i) + '.npy'
    Name_y3before='y3beforePart'+ str(i) + '.npy'
    Name_y2new='y2new'+ str(i) + '.npy'
    Name_y3new='y3new'+ str(i) + '.npy'
    x_train=np.load(NamePartX)
    y_train=np.load(NamePartY) 
    y2before=np.load(Name_y2before) 
    y3before=np.load(Name_y3before) 
    #
    y2forward=model.predict([x_train,y2before,y3before])[0]
    y2before=np.tanh(y2forward)
    y2feedback=Feedback2.predict([x_train,y2before,y3before])
    y2=np.tanh((1-landa)*y2forward+y2feedback)
    y2feedback=Feedback2.predict([x_train,y2,y3before])
    y2new=np.tanh((1-landa)*y2forward+y2feedback)
    #
    y3forward=model.predict([x_train,y2before,y3before])[1]
    y3before=np.tanh(y3forward)
    y3feedback=Feedback3.predict([x_train,y2before,y3before])
    y3=np.tanh((1-landa)*y3forward+y3feedback)
    y3feedback=Feedback3.predict([x_train,y2before,y3])
    y3new=np.tanh((1-landa)*y3forward+y3feedback)
    #  
    y2before=y2 
    y3before=y3
    #
    np.save(Name_y2before,y2before)
    np.save(Name_y3before,y3before)
    np.save(Name_y2new,y2new)
    np.save(Name_y3new,y3new)
#-----------------   
for j in range(100):     
    #print(j)
    model.save('model.h5')
    for i in range(NumParts):
        print('i',j,i)
        NamePartX='TrainPart'+ str(i) + '.npy'
        NamePartY='LabelPart'+ str(i) + '.npy'          
        Name_y2before='y2beforePart'+ str(i) + '.npy'
        Name_y2new='y2new'+ str(i) + '.npy'
        Name_y3new='y3new'+ str(i) + '.npy'
        Name_y3before='y3beforePart'+ str(i) + '.npy'
        x_train=np.load(NamePartX)
        y_train=np.load(NamePartY) 
        y2before=np.load(Name_y2before) 
        y3before=np.load(Name_y3before) 
        y2new=np.load(Name_y2new) 
        y3new=np.load(Name_y3new)
        #
        history = model.fit([x_train,y2before,y3before], [y2new,y3new,y_train],epochs=1,shuffle=True,callbacks= [reduce_LR])
        #
        y2forward=model.predict([x_train,y2before,y3before])[0]
        y2feedback=Feedback2.predict([x_train,y2before,y3before])
        y2=np.tanh((1-landa)*y2forward+y2feedback)
        y2feedback=Feedback2.predict([x_train,y2,y3before])
        y2new=np.tanh((1-landa)*y2forward+y2feedback)  
        #
        y3forward=model.predict([x_train,y2before,y3before])[1]
        y3feedback=Feedback3.predict([x_train,y2before,y3before])
        y3=np.tanh((1-landa)*y3forward+y3feedback)
        y3feedback=Feedback3.predict([x_train,y2before,y3])
        y3new=np.tanh((1-landa)*y3forward+y3feedback)  
        #  
        y2before=y2 
        y3before=y3
        #
        np.save(Name_y2before,y2before)
        np.save(Name_y3before,y3before)  
        np.save(Name_y2new,y2new)
        np.save(Name_y3new,y3new)
#-------------------------------------------------
#test on train data
y2before=0.1*np.random.rand(np.shape(x_train)[0],NumH2) 
y3before=0.1*np.random.rand(np.shape(x_train)[0],NumH3) 
Feedback2 = Model(inputs=model.input,outputs=model.get_layer('f2').output)
Feedback3 = Model(inputs=model.input,outputs=model.get_layer('f3').output)
#   
y2forward=model.predict([x_train,y2before,y3before])[0]
y2before=np.tanh(y2forward)
y2feedback=Feedback2.predict([x_train,y2before,y3before])
y2=np.tanh((1-landa)*y2forward+y2feedback)
#
y3forward=model.predict([x_train,y2before,y3before])[1]
y3before=np.tanh(y3forward)
y3feedback=Feedback3.predict([x_train,y2before,y3before])
y3=np.tanh((1-landa)*y3forward+y3feedback)
#  
y2before=y2 
y3before=y3
#       
m1=model.predict([x_train,y2before,y3before])[2]

for i in range(5):   
    #   
    y2forward=model.predict([x_train,y2before,y3before])[0]
    y2feedback=Feedback2.predict([x_train,y2before,y3before])
    y2=np.tanh((1-landa)*y2forward+y2feedback)
    #
    y3forward=model.predict([x_train,y2before,y3before])[1]
    y3feedback=Feedback3.predict([x_train,y2before,y3before])
    y3=np.tanh((1-landa)*y3forward+y3feedback)
    #  
    y2before=y2 
    y3before=y3
    # 
    

m5=model.predict([x_train,y2before,y3before])[2]
mse1= ((m1 - y_train) ** 2).mean(axis=None)
mse5= ((m5 - y_train) ** 2).mean(axis=None)

#classification error
A1=np.zeros(22216)
A5=np.zeros(22216)
for i in range(22216):
    A1[i]=np.argmax(m1[i,:])
    A5[i]=np.argmax(m5[i,:])
n1=sum(1 for i, j in zip(A1,labels) if i != j) /22216*100  
n5=sum(1 for i, j in zip(A5,labels) if i != j) /22216*100  

#------------------------------------------------------------------------------
import tensorflow as tf
import keras
from keras.models import load_model
model = load_model('model.h5', custom_objects={'tf': tf})
path='D:\Shapar\ShaghayeghUni\AfterPropozal\MyPrograms\EventExtraction\Keras'
PathTest=path+'\TestMat'
PathOut=path+'\TestOut'
import glob
D=glob.glob(PathTest+"/*.mat")
NumTest=len(D)
for i in range(NumTest):
    CB= scipy.io.loadmat(D[i])
    CB_value=CB['CB_context']
    x_train=CB_value
    #---------------------------------------
    #test it 
    y2before=0.1*np.random.rand(np.shape(x_train)[0],NumH2) 
    y3before=0.1*np.random.rand(np.shape(x_train)[0],NumH3) 
    Feedback2 = Model(inputs=model.input,outputs=model.get_layer('f2').output)
    Feedback3 = Model(inputs=model.input,outputs=model.get_layer('f3').output)
    #
    x_train=CB_value
    y2forward=model.predict([x_train,y2before,y3before])[0]
    y2before=np.tanh(y2forward)
    y2feedback=Feedback2.predict([x_train,y2before,y3before])
    y2=np.tanh((1-landa)*y2forward+y2feedback)
    #
    y3forward=model.predict([x_train,y2before,y3before])[1]
    y3before=np.tanh(y3forward)
    y3feedback=Feedback3.predict([x_train,y2before,y3before])
    y3=np.tanh((1-landa)*y3forward+y3feedback)
    #  
    y2before=y2 
    y3before=y3
    #       
    m1=model.predict([x_train,y2before,y3before])[2]  
    '''
    for j in range(5):   
        #   
        y2forward=model.predict([x_train,y2before,y3before])[0]
        y2feedback=Feedback2.predict([x_train,y2before,y3before])
        y2=np.tanh((1-landa)*y2forward+y2feedback)
        #
        y3forward=model.predict([x_train,y2before,y3before])[1]
        y3feedback=Feedback3.predict([x_train,y2before,y3before])
        y3=np.tanh((1-landa)*y3forward+y3feedback)
        #  
        y2before=y2 
        y3before=y3
        # 
    m5=model.predict([x_train,y2before,y3before])[2]
    '''
    y_predict=m1
    #y_predict=m5
    #---------------------------------------
    D2=D[i].replace(PathTest, PathOut)
    scipy.io.savemat(D2, {'lbl': y_predict})
#-----------------------------------------------------------------------------





