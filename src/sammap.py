import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from time import time


dists_D = []
dim_mapped_space = 2

def distance(X, Y):
    #return np.sum(np.abs(X-Y))
    return np.sqrt(np.sum((X-Y)**2))

"""
    This function is the cost function to minimize
    dists_D is the set of distances between the point in the initial space
    d is the set of points in the resulting space in a single row
"""
def stress(d):
    print("************** enter in stress **************")
    global dists_D
    global dim_mapped_space
    good_d=d.reshape((int(len(d)/dim_mapped_space),dim_mapped_space))
    Error = 0
    for i in np.arange(len(good_d[:,0])-1):
        for j in np.arange(i+1, len(good_d[:,0])):
            Error = Error + ((dists_D[i][j-i-1]-distance(good_d[i,:],good_d[j,:]))**2)/dists_D[i][j-i-1]
    print(Error)
    return Error

"""
    This method applies the Sammon's Mapping method on the X variable
    X is a matrix that has observations in rows and features (initial space) in columns
	IT IS IMPORTANT TO MAKE SURE THAT THERE IS NO DOUBLED LINE IN X BEFORE CALLING THIS FUNCTION
"""
def sam(X, dim_result_space = 2):
    #Computation of the distances Dij between the observations in X
    #When we say Dij i is always smaller than j
    global dists_D
    global dim_mapped_space
    dim_mapped_space = dim_result_space
    for i in np.arange(len(X[:,0])-1):
        dists_D.append([])
        for j in np.arange(i+1, len(X[:,0])):
            dists_D[i].append(distance(X[i,:],X[j,:]))
    #initialization of the points in the result space
    Y = np.random.normal(4,1,len(X[:,0])*dim_mapped_space)
    res=minimize(stress, Y, method='Powell', options={'xtol': 1, 'ftol': 1e-1, 'disp':True, 'maxiter':1000})
    return (res.x).reshape((int(len(res.x)/dim_mapped_space),dim_mapped_space))
