"""
In this program I want to use different methods to build attractor in my network

1-load model
2-find outputs
3-AttractorDataPreperation1: find gold outputs and find output of landmark samples (more than 0.5)
4-structure 1: try to make them attractors in an autoassociative net (different structures)
5-use Amini and augmentation methods
6-use tsne and phase plane if succsessful
7-report number of attractors and attractors in train data
8- AttractorDataPreperation2: calculate mean of golds of each class or 
   any better solution to set attractors
9- try steps 4 to 8
if successful:
   10-use this structure as a filter and give its output or hidden layer to a recognition net
   (remind to loop input first)
   11- is it possible to just pass attracted samples tp recognition block?
12-structure 2: manfilod learning(+ multi task learning + metric learning + extract robust embeddings
   same codes for same classes and try to differentiate different classes
13-structure 3: Dehyadegari structure
14-structure 4: diarization method, initialize attractorz and find them later

we have to different methods: 1- reconstruct input  2-just try to exract better embeddings
reconstructing seem to be like generation in GAN  or adversarial training
"""



#-------------------------------------------------
from keras.models import load_model
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
#-------------------------------------------------



#-------------------------------------------------
            # step 1,2,3:
# Load all data:
path='D:\Shapar\ShaghayeghUni\AfterPropozal\MyPrograms\EventExtraction\Keras'

# load matlab output (Just 11 landmarks):
CB_LandamarksTrain_Just11= scipy.io.loadmat(path+'\CB_LandamarksTrain.mat')
x_train=CB_LandamarksTrain_Just11['CB_LandamarksTrain']

CB_BestLandmarksTrain_Just11= scipy.io.loadmat(path+'\CB_BestLandmarksTrain.mat')
y_train=CB_BestLandmarksTrain_Just11['CB_BestLandmarksTrain']
#-------------------------------------------------


#-------------------------
# structure 1:
# DAE model
L=len(x_train[0])
input_tensor = Input(shape=(L,))
x=input_tensor
x = Dense(2000, activation="relu")(x)
x = Dense(2000, activation="relu")(x)
x = Dense(2000, activation="relu")(x)
x = Dense(2000, activation="relu")(x)
x = Dense(2000, activation="relu")(x)
x = Dense(2000, activation="relu")(x)
x = Dense(270, activation="linear")(x)
output = x
model1=Model(input_tensor, output)
model1.summary()

# train parameters:
opt1 = optimizers.Adam(lr=0.00002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model1.compile(optimizer=opt1, loss='mean_squared_error', metrics= ['mean_squared_error'])
reduce_LR = callbacks.ReduceLROnPlateau(monitor='mean_squared_error', factor=0.1, patience=5, verbose=1, mode='max', min_delta=0.0001, cooldown=0, min_lr=0)
tensorboard1 = TensorBoard(path+'/log_dir')
logger = CSVLogger(path+'/training.log')
folder_model="./model"
path_model = os.path.join(folder_model, 'model_epoch_{epoch:02d}.hdf5')
model_checkpoint = callbacks.ModelCheckpoint(filepath= path_model,monitor="mean_squared_error", mode="max", verbose=0, save_best_only=False, save_weights_only=False)    
Early_stop=callbacks.EarlyStopping(monitor='mean_squared_error', min_delta=0.0001, patience=10, verbose=0, mode='auto', baseline=None)

#Train DAE
history = model1.fit(x_train, y_train, batch_size=128,  epochs=40, shuffle=True,verbose=1,
                    callbacks= [model_checkpoint, tensorboard1 , logger , reduce_LR, Early_stop] )
model1.save('modelDae.h5')
#-------------------------------------------------


#----------------------------------------------------------
# Number of attractors in landmraks' positions
Landmarks=np.unique(y_train,axis=0)  
N=1000
data2=Landmarks
for i in range(N):
    OutPredict = model1.predict(data2)
    data2=OutPredict    
LandmarksOutput=OutPredict

mse = (((data2 - Landmarks)**2).mean(axis=1))/810    

NumAttractorOfLandmarks=0;
for i in range (Landmarks.shape[0]) :
    if mse[i]<1e-4:
        NumAttractorOfLandmarks=NumAttractorOfLandmarks+1
print("NumAttractorOfLandmarks",NumAttractorOfLandmarks)  

scipy.io.savemat('LandmarksOutput.mat', {'LandmarksOutput': LandmarksOut})     
scipy.io.savemat('Landmarks.mat', {'Landmarks': Landmarks})      
#------------------------------------------------------------




#----------------------------------------------------------
# Percent of attracted train data
N=10 #increase it
data2=x_train
for i in range(N):
    OutPredict = model1.predict(data2)
    data2=OutPredict    

data3=y_train
for i in range(N):
    OutPredict = model1.predict(data3)
    data3=OutPredict  
    
mse = (((data2 - data3)**2).mean(axis=1))/810    

NumAttractedOfTrain=0;
for i in range (data2.shape[0]) :
    if mse[i]<1e-4:
        NumAttractedOfTrain=NumAttractedOfTrain+1
PercentOfAttraced=NumAttractedOfTrain/data2.shape[0]*100 
#------------------------------------------------------------



#------------------------------------------------------------
# number of attractors in train domain
data_dict = mat73.loadmat(path+'\CB_total_train.mat')
x_train=np.transpose(data_dict['CB_total_train'])
model1 = load_model('modelDae-1.h5')
x_train=x_train.astype('float16')
#---
N=100#increase it
data2=x_train
for i in range(N):
    print(i)
    OutPredict = model1.predict(data2)
    data2=OutPredict   
#--- 
import bisect
NumOtherAtractors=0
Att=LandmarksOutput
for i in range(np.shape(OutPredict)[0]):
    print(i)
    mse = (((OutPredict[i].astype('float16') - Att.astype('float16'))**2).mean(axis=1))
    mse.sort()
    index = bisect.bisect(mse, 0.1)
    if index==0:
        Att=np.concatenate((Att, OutPredict[i:i+1]))
NumAttractors=np.shape(Att)[0]
#-----------------------------------------------------------




#----------------------------------------------------------------------------
#Amini method
L=len(x_train[0])
input_tensor = Input(shape=(L,))
x=input_tensor
x = Dense(500, activation="relu")(x)
x = Dense(500, activation="relu")(x)
x = Dense(500, activation="relu")(x)
x = Dense(500, activation="relu")(x)
x = Dense(810, activation="linear")(x)
output = x
model1=Model(input_tensor, output)
model1.summary()

# train parameters:
opt1 = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model1.compile(optimizer=opt1, loss='mean_squared_error', metrics= ['mean_squared_error'])
reduce_LR = callbacks.ReduceLROnPlateau(monitor='mean_squared_error', factor=0.1, patience=5, verbose=1, mode='max', min_delta=0.0001, cooldown=0, min_lr=0)
tensorboard1 = TensorBoard(path+'/log_dir')
logger = CSVLogger(path+'/training.log')
folder_model="./model"
path_model = os.path.join(folder_model, 'model_epoch_{epoch:02d}.hdf5')
model_checkpoint = callbacks.ModelCheckpoint(filepath= path_model,monitor="mean_squared_error", mode="max", verbose=0, save_best_only=False, save_weights_only=False)    
Early_stop=callbacks.EarlyStopping(monitor='mean_squared_error', min_delta=0.0001, patience=10, verbose=0, mode='auto', baseline=None)

#Train DAE
history = model1.fit(x_train, y_train, batch_size=128,  epochs=1, shuffle=True,verbose=1,
                    callbacks= [model_checkpoint, tensorboard1 , logger , reduce_LR, Early_stop] )

for i in range (50):
    dataOut = model1.predict(x_train)
    data2=np.concatenate((x_train, dataOut), axis=0) 
    data2goal=np.concatenate((y_train, y_train), axis=0) 
    history = model1.fit(x_train, y_train, batch_size=128,  epochs=1, shuffle=True,verbose=1,
                    callbacks= [model_checkpoint, tensorboard1 , logger , reduce_LR, Early_stop] )
model1.save('modelَAmini.h5')
#----------------------------------------------------------------------------
    





#----------------------------------------------------------------------------
path='D:\Shapar\ShaghayeghUni\AfterPropozal\MyPrograms\EventExtraction\Keras'
#Amini method + partition
#input
LenPart=np.load('LenPart.npy')
NumParts=np.load('NumParts.npy')
i=0;
NamePartX='TrainLandmarkPart'+ str(i) + '.npy'
NamePartY='BestLandmarkPart'+ str(i) + '.npy' 
x_train=np.load(NamePartX)
y_train=np.load(NamePartY)

L=len(x_train[0])
input_tensor = Input(shape=(L,))
x=input_tensor
x = Dense(2000, activation="relu")(x)
x = Dense(2000, activation="relu")(x)
x = Dense(2000, activation="relu")(x)
x = Dense(2000, activation="relu")(x)
x = Dense(2000, activation="relu")(x)
x = Dense(2000, activation="relu")(x)
x = Dense(270, activation="linear")(x)
output = x
model1=Model(input_tensor, output)
model1.summary()

# train parameters:
opt1 = optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model1.compile(optimizer=opt1, loss='mean_squared_error', metrics= ['mean_squared_error'])
reduce_LR = callbacks.ReduceLROnPlateau(monitor='mean_squared_error', factor=0.1, patience=5, verbose=1, mode='max', min_delta=0.0001, cooldown=0, min_lr=0)
tensorboard1 = TensorBoard(path+'/log_dir')
logger = CSVLogger(path+'/training.log')
folder_model="./model"
path_model = os.path.join(folder_model, 'model_epoch_{epoch:02d}.hdf5')
model_checkpoint = callbacks.ModelCheckpoint(filepath= path_model,monitor="mean_squared_error", mode="max", verbose=0, save_best_only=False, save_weights_only=False)    
Early_stop=callbacks.EarlyStopping(monitor='mean_squared_error', min_delta=0.0001, patience=10, verbose=0, mode='auto', baseline=None)

#Train DAE
history = model1.fit(x_train, y_train, batch_size=128,  epochs=1, shuffle=True,verbose=1,
                    callbacks= [model_checkpoint, tensorboard1 , logger , reduce_LR, Early_stop] )

for j in range (50):
    for i in range(NumParts):
        print('i',j,i)
        NamePartX='TrainLandmarkPart'+ str(i) + '.npy'
        NamePartY='BestLandmarkPart'+ str(i) + '.npy'          
        x_train=np.load(NamePartX)
        y_train=np.load(NamePartY) 
        dataOut = model1.predict(x_train)
        data2=np.concatenate((x_train, dataOut), axis=0) 
        data2goal=np.concatenate((y_train, y_train), axis=0) 
        history = model1.fit(x_train, y_train, batch_size=128,  epochs=1, shuffle=True,verbose=1,
                    callbacks= [model_checkpoint, tensorboard1 , logger , reduce_LR, Early_stop] )
model1.save('Amini.h5')
#----------------------------------------------------------------------------




#----------------------------------------------------------------------------
#Augmentation method
#talim ba dor zadan
L=len(x_train[0])
input_tensor = Input(shape=(L,))
x=input_tensor
x = Dense(500, activation="relu")(x)
x = Dense(500, activation="relu")(x)
x = Dense(500, activation="relu")(x)
x = Dense(500, activation="relu")(x)
x = Dense(810, activation="linear")(x)
output = x
model1=Model(input_tensor, output)
model1.summary()

# train parameters:
opt1 = optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model1.compile(optimizer=opt1, loss='mean_squared_error', metrics= ['mean_squared_error'])
reduce_LR = callbacks.ReduceLROnPlateau(monitor='mean_squared_error', factor=0.1, patience=5, verbose=1, mode='max', min_delta=0.0001, cooldown=0, min_lr=0)
tensorboard1 = TensorBoard(path+'/log_dir')
logger = CSVLogger(path+'/training.log')
folder_model="./model"
path_model = os.path.join(folder_model, 'model_epoch_{epoch:02d}.hdf5')
model_checkpoint = callbacks.ModelCheckpoint(filepath= path_model,monitor="mean_squared_error", mode="max", verbose=0, save_best_only=False, save_weights_only=False)    
Early_stop=callbacks.EarlyStopping(monitor='mean_squared_error', min_delta=0.0001, patience=10, verbose=0, mode='auto', baseline=None)


data=x_train
dataOut=y_train
AugNum=4
for i in range (AugNum):
    dataAug=x_train+ (0.05*(np.random.rand(np.shape(x_train)[0],np.shape(x_train)[1])-2)) 
    data=np.concatenate((data, dataAug), axis=0)
    dataOut=np.concatenate((dataOut, y_train), axis=0)

#Train DAE
history = model1.fit(data, dataOut, batch_size=128,  epochs=30, shuffle=True,verbose=1,
                    callbacks= [model_checkpoint, tensorboard1 , logger , reduce_LR, Early_stop] )
model1.save('modelAug.h5')
#----------------------------------------------------------------------------
    
  

#----------------------------------------------------------------------------
#Augmentation method + RAM problem
#talim ba dor zadan
L=len(x_train[0])
input_tensor = Input(shape=(L,))
x=input_tensor
x = Dense(2000, activation="relu")(x)
x = Dense(2000, activation="relu")(x)
x = Dense(2000, activation="relu")(x)
x = Dense(2000, activation="relu")(x)
x = Dense(2000, activation="relu")(x)
x = Dense(2000, activation="relu")(x)
x = Dense(270, activation="linear")(x)
output = x
model1=Model(input_tensor, output)
model1.summary()

# train parameters:
opt1 = optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model1.compile(optimizer=opt1, loss='mean_squared_error', metrics= ['mean_squared_error'])
reduce_LR = callbacks.ReduceLROnPlateau(monitor='mean_squared_error', factor=0.1, patience=5, verbose=1, mode='max', min_delta=0.0001, cooldown=0, min_lr=0)
tensorboard1 = TensorBoard(path+'/log_dir')
logger = CSVLogger(path+'/training.log')
folder_model="./model"
path_model = os.path.join(folder_model, 'model_epoch_{epoch:02d}.hdf5')
model_checkpoint = callbacks.ModelCheckpoint(filepath= path_model,monitor="mean_squared_error", mode="max", verbose=0, save_best_only=False, save_weights_only=False)    
Early_stop=callbacks.EarlyStopping(monitor='mean_squared_error', min_delta=0.0001, patience=10, verbose=0, mode='auto', baseline=None)


NamePartX=path+'/x_train'+ str(0)
np.save(NamePartX,x_train)
AugNum=4
for i in range (AugNum):
    dataAug=x_train+ (0.05*(np.random.rand(np.shape(x_train)[0],np.shape(x_train)[1])-2)) 
    NamePartX=path+'/x_train'+ str(i+1)
    np.save(NamePartX,dataAug)

NamePartX=path+'/x_train'+ str(0)
np.save(NamePartX,x_train)
AugNum=4
for i in range (AugNum):
    dataAug=x_train+ (0.05*(np.random.rand(np.shape(x_train)[0],np.shape(x_train)[1])-2)) 
    NamePartX=path+'/x_train'+ str(i+1)
    np.save(NamePartX,dataAug)

#Train DAE
for j in range(50):
  for i in range(5):
      NamePartX=path+'/x_train'+ str(i)+'.npy'
      del x_train
      x_train=np.load(NamePartX)
      model1.fit(x_train, y_train, batch_size=128,  epochs=1, shuffle=True,verbose=1 )
model1.save('modelAug.h5')
#----------------------------------------------------------------------------



#----------------------------------------------------------------------------
# with balanced data: 
L=len(x_train[0])
input_tensor = Input(shape=(L,))
x=input_tensor
x = Dense(500, activation="relu")(x)
x = Dense(500, activation="relu")(x)
x = Dense(500, activation="relu")(x)
x = Dense(500, activation="relu")(x)
x = Dense(270, activation="linear")(x)
output = x
model1=Model(input_tensor, output)
model1.summary()

# weights:
LandmarksLabels_Just11 = scipy.io.loadmat(path+'\LandmarksLabels_Just11.mat')
labels=LandmarksLabels_Just11['LandmarksLabels_Just11'][0]
"""
W=np.zeros(11)
for j in range(11):
    W[j]=sum(1 for i in range(22216) if labels[i]== j)
"""    
# train parameters:
opt1 = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model1.compile(optimizer=opt1, loss='mean_squared_error', metrics= ['mean_squared_error'])
reduce_LR = callbacks.ReduceLROnPlateau(monitor='mean_squared_error', factor=0.1, patience=5, verbose=1, mode='max', min_delta=0.0001, cooldown=0, min_lr=0)
tensorboard1 = TensorBoard(path+'/log_dir')
logger = CSVLogger(path+'/training.log')
folder_model="./model"
path_model = os.path.join(folder_model, 'model_epoch_{epoch:02d}.hdf5')
model_checkpoint = callbacks.ModelCheckpoint(filepath= path_model,monitor="mean_squared_error", mode="max", verbose=0, save_best_only=False, save_weights_only=False)    
Early_stop=callbacks.EarlyStopping(monitor='mean_squared_error', min_delta=0.0001, patience=10, verbose=0, mode='auto', baseline=None)
#-----------
# class weight
from sklearn.utils import class_weight
class_weights = dict(zip(np.unique(labels), class_weight.compute_class_weight('balanced',np.unique(labels),labels)))
history = model1.fit(x_train, y_train, batch_size=128,  epochs=40, class_weight=class_weights,shuffle=True,verbose=1,
callbacks= [model_checkpoint, tensorboard1 , logger , reduce_LR, Early_stop] )
#-----------
# sample weight
sample_weight = np.ones(shape=(len(x_train),))
for i in range(11):
    sample_weight[labels == i] = class_weights[i]
history = model1.fit(x_train, y_train, batch_size=128,  epochs=40,
 sample_weight=sample_weight ,shuffle=True,verbose=1,
callbacks= [model_checkpoint, tensorboard1 , logger , reduce_LR, Early_stop] )
model1.save('modeDaeSampleWeight.h5')
#-----------
# random sampling (to downsample rich classes)
ClassNumbers=np.zeros(11)
for j in range(11):
    ClassNumbers[j]=sum(1 for i in range(22216) if labels[i]== j)
NumMin=int(np.min(ClassNumbers))


from random import sample 

def RandomBalanceBatch(x_train,y_train,ClassNumbers,NumMin):
    I=0
    BalancedBatch=np.zeros((1,810))
    BalancedOut=np.zeros((1,810))
    for i in range(11):
        A=list(range(int(ClassNumbers[i])))
        A=[x+I for x in A]
        randomlist=sample(A,NumMin)
        BB=x_train[randomlist,:]
        BalancedBatch=np.concatenate((BalancedBatch,BB),axis=0)
        BB=y_train[randomlist,:]
        BalancedOut=np.concatenate((BalancedOut,BB),axis=0)
        I=int(I+ClassNumbers[i])
    BalancedBatch=np.delete(BalancedBatch, 0, 0)
    BalancedOut=np.delete(BalancedOut, 0, 0)
    return(BalancedBatch,BalancedOut)

for i in range(4000):
    print(i)
    BalancedBatch,BalancedOut=RandomBalanceBatch(x_train,y_train,ClassNumbers,NumMin)
    history = model1.fit(BalancedBatch, BalancedOut,  epochs=1, shuffle=True,verbose=1,
	callbacks= [model_checkpoint, tensorboard1 , logger , reduce_LR, Early_stop] )
#----------------------------------------------------------------------------


    
#----------------------------------------------------------------------------
# draw tsne before and after tranformation with autoassociative network
LandmarksLabels_Just11 = scipy.io.loadmat(path+'\LandmarksLabels_Just11.mat')
labels=LandmarksLabels_Just11['LandmarksLabels_Just11'][0]

from sklearn import datasets
from sklearn.decomposition import PCA
from TsneSpk import TsneSpk

pca = PCA(n_components=50)
pca.fit(x_train) 
x_train_pca=pca.transform(x_train)
NumOutput=11
TsneSpk(x_train_pca,labels,NumOutput)

dataOut = model1.predict(x_train)
pca = PCA(n_components=50)
pca.fit(dataOut) 
dataOut_pca=pca.transform(dataOut)
NumOutput=11
TsneSpk(dataOut_pca,labels,NumOutput)

x0=x_train
for i in range(100):
    print(i)
    dataOut = model1.predict(x0)
    x0=dataOut
pca = PCA(n_components=50)
pca.fit(dataOut) 
dataOut_pca=pca.transform(dataOut)
NumOutput=11
TsneSpk(dataOut_pca,labels,NumOutput)
#----------------------------------------------------------------------------





#----------------------------------------------------------------------------
# draw attraction path and basin of attraction
"""
import numpy as np
import matplotlib.pyplot as plt

model1 = load_model('modelDae-1.h5')
pca = PCA(n_components=2)
pca.fit(x_train) 
x_train_pca=pca.transform(x_train)


X, Y = np.meshgrid(np.linspace(-30, 30, 200), np.linspace(-30, 30, 200))
u, v = np.zeros_like(X), np.zeros_like(X)
NI, NJ = X.shape

#attraction path and phase space
data2=np.random.rand(1,2)
for i in range(NI):
    print(i)
    for j in range(NJ):
        data2[0][0]=X[i, j]
        data2[0][1]=Y[i, j]
        x_train_pcaInv=pca.inverse_transform(data2)
        OutPredict = model1.predict(x_train_pcaInv)
        out_pca=pca.transform(OutPredict)
        u[i,j]=out_pca[0][0]-X[i, j]
        v[i,j]=out_pca[0][1]-Y[i, j]
plt.streamplot(X, Y, u, v)
plt.quiver(X, Y, u, v, units='width')
plt.axis('square')
plt.axis([-30, 30, -30, 30])
plt.title('Phase Plane')
"""
#----------------------------------------------------------------------------



#----------------------------------------------------------------------------
# use this attractor network (autoassociative network) to filter and then recognize landmarks
#first: recognize before filtering:
path='D:\Shapar\ShaghayeghUni\AfterPropozal\MyPrograms\EventExtraction\Keras'
data_dict = mat73.loadmat(path+'\CB_total_train.mat')
x_train=data_dict['CB_total_train']

LBL_total_train= scipy.io.loadmat(path+'\LBL_total_train.mat')
LBL_train=LBL_total_train['LBL_total_train']

L=len(x_train[0])
input_tensor = Input(shape=(L,))
x=input_tensor
x = Dense(1000, activation="relu")(x)
x = Dense(1000, activation="relu")(x)
x = Dense(1000, activation="relu")(x)
x = Dense(1000, activation="relu")(x)
x = Dense(1000, activation="relu")(x)
x = Dense(103, activation="linear")(x)
output = x
modelRec=Model(input_tensor, output)
modelRec.summary()

# train parameters:
opt1 = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
modelRec.compile(optimizer=opt1, loss='mean_squared_error', metrics= ['mean_squared_error'])
reduce_LR = callbacks.ReduceLROnPlateau(monitor='mean_squared_error', factor=0.1, patience=5, verbose=1, mode='max', min_delta=0.0001, cooldown=0, min_lr=0)
tensorboard1 = TensorBoard(path+'/log_dir')
logger = CSVLogger(path+'/training.log')
folder_model="./model"
path_model = os.path.join(folder_model, 'model_epoch_{epoch:02d}.hdf5')
model_checkpoint = callbacks.ModelCheckpoint(filepath= path_model,monitor="mean_squared_error", mode="max", verbose=0, save_best_only=False, save_weights_only=False)    
Early_stop=callbacks.EarlyStopping(monitor='mean_squared_error', min_delta=0.0001, patience=10, verbose=0, mode='auto', baseline=None)

# Without attractor network filtering:
history = modelRec.fit(x_train, LBL_train, batch_size=128,  epochs=40, shuffle=True,verbose=1,
                    callbacks= [model_checkpoint, tensorboard1 , logger , reduce_LR, Early_stop] )
modelRec.save('modelRec.h5')



OutPredict1 = modelRec.predict(x_train)
mse1= ((OutPredict1  - LBL_train) ** 2).mean(axis=None)
trainFiltered = modelDae.predict(x_train)
OutPredict2 = modelRec.predict(trainFiltered)
mse2= ((OutPredict2  - LBL_train) ** 2).mean(axis=None)

'''
# With one time attractor network filtering:
#modelDae= load_model('modelDae.h5')
modelDae= load_model('modelAug.h5')
trainFiltered = modelDae.predict(x_train)
model1.compile(optimizer=opt1, loss='mean_squared_error', metrics= ['mean_squared_error'])
history = model1.fit(trainFiltered, LBL_train, batch_size=128,  epochs=40, shuffle=True,verbose=1,
                    callbacks= [model_checkpoint, tensorboard1 , logger , reduce_LR, Early_stop] )
model1.save('modelRecAfterFiltering1.h5')
OutPredict2 = model1.predict(trainFiltered)
# = model1.predict(trainFiltered)

# With N time attractor network filtering:
N=2
trainFiltered = x_train
for i in range(N):
    trainFiltered = modelDae.predict(trainFiltered)
model1.compile(optimizer=opt1, loss='mean_squared_error', metrics= ['mean_squared_error'])
history = model1.fit(trainFiltered, labels_caLBL_traint, batch_size=128,  epochs=40, shuffle=True,verbose=1,
                    callbacks= [model_checkpoint, tensorboard1 , logger , reduce_LR, Early_stop] )

#use weights:    
from sklearn.utils import class_weight
class_weights = dict(zip(np.unique(labels), class_weight.compute_class_weight('balanced',np.unique(labels),labels)))
model1.compile(optimizer=opt1, loss='mean_squared_error', metrics= ['mean_squared_error'])
history = model1.fit(trainFiltered, LBL_train, batch_size=128,  epochs=40,class_weight=class_weights, shuffle=True,verbose=1,
                    callbacks= [model_checkpoint, tensorboard1 , logger , reduce_LR, Early_stop] )
model1.save('modelRecAfterFiltering200.h5')
OutPredict4 = model1.predict(trainFiltered)
'''
#----------------------------------------------------------------------------





#------------------------------------------------------------------------------
import keras
from keras.models import load_model
modelDae = load_model(path+'modelDae.h5')
modelRec1 = load_model(path+'modelRec1.h5')
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
    #-----
    trainFiltered = modelDae.predict(x_train)
    y_predict = modelRec1.predict(trainFiltered)
    #---------------------------------------
    D2=D[i].replace(PathTest, PathOut)
    scipy.io.savemat(D2, {'lbl': y_predict})
#-----------------------------------------------------------------------------
    
    