from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA as sklearnPCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sammap

algorithms = ["Isometric Feature Mapping", "Principal Component Analysis", "Local Linear Embedding", "Multidimensional Scaling"]
features_to_code = ['Gender', 'Student', 'Married', 'Ethnicity']

def code_feature(obsv_feature):
    result = {}
    code = 0
    for row in obsv_feature:
        if not row in result:
            result[row]=code
            code = code+1
    return result 

data = pd.read_csv("../datasets/credit.csv")

for feature in features_to_code:
    codage_feature = code_feature(data[feature].values)
    for val_to_code in codage_feature:
        data[feature].values[data[feature].values[:]==val_to_code]=codage_feature[val_to_code]

isomap = Isomap(n_components=2)
sklearn_pca = sklearnPCA(n_components=2)
lle = LocallyLinearEmbedding(n_components=2)
mds = MDS(n_components=2)

i = 0
for algo in (isomap, sklearn_pca, lle, mds):
    Y_sklearn = algo.fit_transform(data.values[:,1:])
    ax=plt.subplot(231+i)
    ax.scatter(Y_sklearn[:,0], Y_sklearn[:,1], marker='o', color='green')
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
ax.scatter(Y_sammap[:,0], Y_sammap[:,1] , marker='o', color='green')
ax.set_title('Sammon\'s mapping')
plt.show()


