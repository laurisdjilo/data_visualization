from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA as sklearnPCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sammap

algorithms = ["Isometric Feature Mapping", "Principal Component Analysis", "Local Linear Embedding", "Multidimensional Scaling"]
digitcolors = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
colors = []

def genColors (nberColors):
    color = '#000000'
    for i in range(nberColors):
        while color in colors:
            color = '#'
            for j in range(6):
                color = color + digitcolors[np.random.randint(0,15)]
        colors.append(color)


data = pd.read_csv("../datasets/mtcars.csv")
genColors(len(data.values[:,0]))

isomap = Isomap(n_components=2)
sklearn_pca = sklearnPCA(n_components=2)
lle = LocallyLinearEmbedding(n_components=2)
mds = MDS(n_components=2)

i = 0
for algo in (isomap, sklearn_pca, lle, mds):
    Y_sklearn = algo.fit_transform(data.values[:,1:])
    ax=plt.subplot(231+i)
    for dot in range(len(data.values[:,0])):
        ax.scatter(Y_sklearn[dot,0], Y_sklearn[dot,1], marker='o', color=colors[dot], label=data.values[dot,0])
    #ax.scatter(Y_sklearn[:,0], Y_sklearn[:,1], marker='o', color='green')
    ax.set_title(algorithms[i])
    i = i+1

matrix = data.values[:,:]
i=0
sup_i = len(matrix)-1
while i < sup_i:
    j=i+1
    sup_j = len(matrix)
    while j >= i+1 and j < sup_j:
        if np.all((matrix[i,1:]-matrix[j,1:])==0):
            matrix=np.delete(matrix,j,0)
            sup_i = sup_i-1
            sup_j = sup_j-1
            j=j-1
        j = j+1
    i = i+1
Y_sammap = sammap.sam(matrix[:,1:],2)

ax=plt.subplot(231+4)
for dot in range(len(matrix[:,0])):
	ax.scatter(Y_sammap[dot,0], Y_sammap[dot,1], marker='o', color=colors[dot], label=matrix[dot,0])
ax.set_title('Sammon\'s mapping')
plt.legend(bbox_to_anchor=(2.2, 2.55), loc=2)
plt.show()


