

# ----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)

from random import shuffle



# ----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X,Y,NumSpks,title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    #ax = plt.subplot(111)
    #colors = cm.rainbow(np.linspace(0, 1, 20))
    for i in range(X.shape[0]):
        #plt.scatter(X[i, 0], X[i, 1], color=colors[y[i]]) 
        plt.scatter(X[i, 0], X[i, 1], color=plt.cm.Set1(Y[i] / NumSpks)) 
        """
    for i in range(X.shape[0]):
        #plt.scatter(X[i, 0], X[i, 1], color=plt.cm.Set1(y[i] / NumSpks)) 
        plt.scatter(X[i, 0], X[i, 1], color=plt.cm.Set1(y[i]  / NumSpks)) 
        """
    
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)    
# ----------------------------------------------------------------------



    
# ----------------------------------------------------------------------
def TsneSpk(embedings,labels,NumSpks):
    # ----------------------
    '''
    #randomly select spks!!!!!
    n_samples, n_features = embedings.shape
    r = [i for i in range(np.max(labels)+1)]
    shuffle(r)
    #r=r[0:NumSpks]
    
    X=np.zeros((1,n_features))
    Y=[]
    for i in range(NumSpks):
        y = [labels[j] for j in range(len(labels)) if labels[j] == r[i]]
        y = np.array(y)
        x=[embedings[j] for j in range(len(labels)) if labels[j] == r[i]]
        x=np.array(x)
        X=np.append(X,x,axis=0)
        y[:]=i
        Y=np.append(Y,y)
    X=np.delete(X, 0, 0)
    #save selected speakers to compare methods!!!!!
    '''
    # ---------------------
    # t-SNE embedding of the digits dataset
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(embedings)
    plot_embedding(X_tsne,labels,NumSpks,title='Tsne plot')
# ----------------------------------------------------------------------



#(n_components=2, *, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, 
#n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean',
# init='random' or 'pca', random_state=None, method='barnes_hut', angle=0.5, n_jobs=None)[source]¶

