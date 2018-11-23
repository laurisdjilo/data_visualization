from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA as sklearnPCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sammap

algorithms = ["Isometric Feature Mapping", "Principal Component Analysis", "Local Linear Embedding", "Multidimensional Scaling"]
features_to_code = [1,3,5,6,8,10,12,14,15,16,17,18,19,20]
nbre_lines = 400

def code_feature(obsv_feature):
	result = {}
	code = 0
	for row in obsv_feature:
		if not str(row) in result:
			result[str(row)]=code
			code = code+1
	return result 

data = pd.read_csv("../datasets/chemicals_in_Cosmetics.csv")
values = data.values[:nbre_lines,:]
boolean_missings = data.isnull().values[:nbre_lines,:]
values[boolean_missings]=-200
values = values.T
for feature in features_to_code:
	codage_feature = code_feature(values[feature])
	for val_to_code in codage_feature:
		values[feature][[str(temp)==val_to_code for temp in values[feature]]]=codage_feature[val_to_code]
values = values.T
print(values)

isomap = Isomap(n_components=2)
sklearn_pca = sklearnPCA(n_components=2)
lle = LocallyLinearEmbedding(n_components=2)
mds = MDS(n_components=2)

i = 0
for algo in (isomap, sklearn_pca, lle, mds):
	Y_sklearn = algo.fit_transform(values)
	ax=plt.subplot(231+i)
	ax.scatter(Y_sklearn[:,0], Y_sklearn[:,1], marker='o', color='green')
	ax.set_title(algorithms[i])
	i = i+1

matrix = values
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

ax=plt.subplot(231+4)
ax.scatter(Y_sammap[:,0], Y_sammap[:,1] , marker='o', color='green')
ax.set_title('Sammon\'s mapping')
plt.show()
