# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 16:35:33 2021

@author: user
"""



# ----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)




# ----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X,Y,NumSpks,title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()

    for i in range(233):
        plt.scatter(X[i, 0], X[i, 1], color=plt.cm.Set1(Y[i] / NumSpks)) 

    
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)    
# ----------------------------------------------------------------------



    
# ----------------------------------------------------------------------
def TsneSpk(embedings,labels,NumSpks):

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

