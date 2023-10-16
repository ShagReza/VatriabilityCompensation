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
              loss_weights={'v6': 0.8, 'v4': 0.2},
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
model.layers[6].set_weights(WeightL6)
#------------------------------------------------------------------------------
# To select Best landmarks again
x_train=SelectedLandmarks
y_train=OutLandmarks
for k in range(20):
    #------------------------------------------------------------------------------    
    # Find best landmarks
    FinalLandmarkCode = scipy.io.loadmat(path+'\FinalLandmarkCode.mat')
    LandmarkCode=FinalLandmarkCode['FinalLandmarkCode']
    x_out=model.predict(x_train)[1]
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
            #----------------------
            SelectedLandmarks=np.append(SelectedLandmarks, x_train[LabelI_index], axis=0)
            OutLandmarks=np.append(OutLandmarks, y_train[LabelI_index], axis=0)
            repeats_array = np.tile(Best, (np.shape(x_out_i)[0], 1))
            BestLandmraks=np.append(BestLandmraks, repeats_array, axis=0)
    SelectedLandmarks=np.delete(SelectedLandmarks, 1, 0)
    BestLandmraks=np.delete(BestLandmraks, 1, 0)
    OutLandmarks=np.delete(OutLandmarks, 1, 0)
    # Find embeding layer's output
    intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer('v4').output)
    EmbedingOfBestLandmarks = intermediate_layer_model.predict([BestLandmraks])
    #------------------------------------------------------------------------------
        # train on Best landmarks
    history = model.fit(SelectedLandmarks,[EmbedingOfBestLandmarks,OutLandmarks], batch_size=128,  epochs=1, shuffle=True,verbose=1)




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

intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer('v4').output)
intermediate_output = intermediate_layer_model.predict([LandmarkI5])
pca = PCA(n_components=50)
pca.fit(intermediate_output) 
LandmrakI_pca=pca.transform(intermediate_output)
NumOutput=5
TsneSpk(LandmrakI_pca,Labels5,NumOutput)
#------------------------------------------------------------------------------



