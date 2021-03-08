import numpy as np
import pandas as pd
import math
import sys
import datetime
import time
import random
import scipy as sp
from sklearn.cluster import KMeans

def get_cartesian_from_barycentric(b):
    t = np.transpose(np.array([[0,0],[1,0],[0.5,sqrt(3)/2]])) # Triangle
    return t.dot(b)

def findClusters(df):
    distancedict = {}
    lptracks = df.shape[0]
    print (lptracks)
    A = []
    X = []
    asq = [[(1)] * lptracks]
    bsq = [[1] * lptracks]
    csq = [[1] * lptracks]
    C1 = df['C_1']
    C2 = df['C_2']
    C3 = df['C_3']
    for p in range(0,lptracks):
        P1 = [C1.iloc[p]] * lptracks
        P2 = [C2.iloc[p]] * lptracks
        P3 = [C3.iloc[p]] * lptracks
        Q1 = (C1).to_numpy()
        Q2 = (C2).to_numpy()
        Q3 = (C3).to_numpy()
        D = -(Q2 - P2)*(Q3 - P3)*asq - (Q1 - P1)*(Q3-P3)*bsq - (Q1 - P1)*(Q2 - P2)*csq
        D = D[0]
        mask1 = D<0.09
        targetCluster = df.iloc[p]['cluster.ID']
        mask2 = df['cluster.ID'] == targetCluster
        mask = mask1 & mask2
        idx = np.where (mask==True)[0]
        testdf=pd.DataFrame()
        testdf['difference'] = D[idx]
        testdf['cluster'] = df['cluster.ID'][idx].tolist()
        testdf.to_csv('file.csv')
        D[mask] = 1
        D[~mask] = 0
        A.append(D.tolist())
    A = np.asarray(A)
    D = np.diag(np.sum(A, axis=0))
    # graph laplacian
    L = np.subtract(D,A)
    # eigenvalues and eigenvectors
    vals, vecs = np.linalg.eig(L)
    # sort these based on the eigenvalues
    vecs = vecs[:,np.argsort(vals)].real
    vals = vals[np.argsort(vals)].real
    print(vals)
 
    #limiteigen = 0.1 * vals[vals.size-1]
    #count = len([eigen for eigen in vals if eigen < limiteigen])
    if (vals.size > 1):
        limiteigen = 0.1 * vals[vals.size-1]
        if (limiteigen > 0):
            count = len([eigen for eigen in vals if abs(eigen) < limiteigen])
        else:
            count = vals.size

        # kmeans on first three vectors with nonzero eigenvalues
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(vecs[:,0:count-1])
        colors = kmeans.labels_
 
    else:
        print("There is just one cluster")
    return 

def main():
    df = pd.read_csv('../toUFl.20210222/channel.flow/data/channel_clustering_11-4-20.dat', header=0, names=['C_1', 'C_2', 'C_3', 'eta_1', 'lambda_1', 'eta_4', 'eta_3', 'case.no', 'cluster.ID', 'x', 'y', 'z'])
    findClusters(df)

if __name__=="__main__":
    main()

