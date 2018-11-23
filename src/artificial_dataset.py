from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA as sklearnPCA
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from sklearn import datasets
import sammap


n_points = 100
X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
fig = plt.figure()
ax = plt.subplot(231,projection='3d')

ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)

algorithms = ["Isometric Feature Mapping", "Principal Component Analysis", "Local Linear Embedding", "Multidimensional Scaling"]

isomap = Isomap(n_components=2)
sklearn_pca = sklearnPCA(n_components=2)
lle = LocallyLinearEmbedding(n_components=2)
mds = MDS(n_components=2)

i = 1
for algo in (isomap, sklearn_pca, lle, mds):
    Y_sklearn = algo.fit_transform(X)
    ax=plt.subplot(231+i)
    ax.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=color, cmap=plt.cm.Spectral)
    ax.set_title(algorithms[i-1])
    i = i+1

matrix = X[:,:]
i=0
sup_i = len(matrix)-1
while i < sup_i:
    j=i+1
    sup_j = len(matrix)
    while j >= i+1 and j < sup_j:
        if np.all((matrix[i,:]-matrix[j,:])==0):
            matrix=np.delete(matrix,j,0)
            sup_i = sup_i-1
            sup_j = sup_j-1
            j=j-1
        j = j+1
    i = i+1
Y_sammap = sammap.sam(matrix[:,:],2)

ax=plt.subplot(231+5)
ax.scatter(Y_sammap[:,0], Y_sammap[:,1] , c=color, cmap=plt.cm.Spectral)
ax.set_title('Sammon\'s mapping')

plt.show()
