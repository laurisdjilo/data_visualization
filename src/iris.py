from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA as sklearnPCA
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import sammap

algorithms = ["Isometric Feature Mapping", "Principal Component Analysis", "Local Linear Embedding", "Multidimensional Scaling"]

data = pd.read_csv("../datasets/iris.csv")

sklear_pca_3d = sklearnPCA(n_components=3)
isomap = Isomap(n_components=2)
sklearn_pca = sklearnPCA(n_components=2)
lle = LocallyLinearEmbedding(n_components=2)
mds = MDS(n_components=2)


indices_setosa = []
indices_versicolor = []
indices_virginica = []
ind = 0
for row in data.values[:,4]:
	if row=='Iris-setosa':
	    indices_setosa.append(ind)
	elif row=='Iris-versicolor':
	    indices_versicolor.append(ind)
	else:
	    indices_virginica.append(ind)
	ind = ind+1
Y_sklearn = sklear_pca_3d.fit_transform(data.values[:,0:4])

fig = plt.figure()
ax = plt.subplot(231,projection='3d')

ax.scatter3D(np.take(Y_sklearn[:,0], indices_setosa),np.take(Y_sklearn[:,1], indices_setosa),np.take(Y_sklearn[:,2], indices_setosa), marker='o', color='green', label='Iris-Setosa')
ax.scatter3D(np.take(Y_sklearn[:,0], indices_versicolor),np.take(Y_sklearn[:,1], indices_versicolor),np.take(Y_sklearn[:,2], indices_versicolor), marker='o', color='red', label= 'Iris-Versicolor')
ax.scatter3D(np.take(Y_sklearn[:,0], indices_virginica),np.take(Y_sklearn[:,1], indices_virginica),np.take(Y_sklearn[:,2], indices_virginica), marker='o', color='blue', label='Iris-Virginica')

i = 1
for algo in (isomap, sklearn_pca, lle, mds):
    Y_sklearn = algo.fit_transform(data.values[:,0:4])
    ax=plt.subplot(231+i)
    ax.scatter(np.take(Y_sklearn[:,0], indices_setosa),np.take(Y_sklearn[:,1], indices_setosa), marker='o', color='green', label='Iris-Setosa')
    ax.scatter(np.take(Y_sklearn[:,0], indices_versicolor),np.take(Y_sklearn[:,1], indices_versicolor), marker='o', color='red', label= 'Iris-Versicolor')
    ax.scatter(np.take(Y_sklearn[:,0], indices_virginica),np.take(Y_sklearn[:,1], indices_virginica), marker='o', color='blue', label='Iris-Virginica')
    ax.set_title(algorithms[i-1])
    i = i+1

matrix = data.values[:,:]
i=0
sup_i = len(matrix)-1
while i < sup_i:
    j=i+1
    sup_j = len(matrix)
    while j >= i+1 and j < sup_j:
        if np.all((matrix[i,0:4]-matrix[j,0:4])==0):
            matrix=np.delete(matrix,j,0)
            sup_i = sup_i-1
            sup_j = sup_j-1
            j=j-1
        j = j+1
    i = i+1
Y_sammap = sammap.sam(matrix[:,0:4],2)

indices_setosa = []
indices_versicolor = []
indices_virginica = []
ind = 0
for row in matrix[:,4]:
    if row=='Iris-setosa':
        indices_setosa.append(ind)
    elif row=='Iris-versicolor':
        indices_versicolor.append(ind)
    else:
        indices_virginica.append(ind)
    ind = ind+1
ax=plt.subplot(231+5)
ax.scatter(np.take(Y_sammap[:,0], indices_setosa),np.take(Y_sammap[:,1], indices_setosa), marker='o', color='green', label='Iris-Setosa')
ax.scatter(np.take(Y_sammap[:,0], indices_versicolor),np.take(Y_sammap[:,1], indices_versicolor), marker='o', color='red', label= 'Iris-Versicolor')
ax.scatter(np.take(Y_sammap[:,0], indices_virginica),np.take(Y_sammap[:,1], indices_virginica), marker='o', color='blue', label='Iris-Virginica')
ax.set_title('Sammon\'s mapping')

plt.legend(bbox_to_anchor=(2.2,2.55), loc=2)
plt.show()
