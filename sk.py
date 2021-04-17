import numpy as np
import pandas as pd
import math
import sys
import datetime
import time
import random
import scipy as sp
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import umap

def get_cartesian_from_barycentric(b):
    t = np.transpose(np.array([[0,0],[1,0],[0.5,sqrt(3)/2]])) # Triangle
    return t.dot(b)

def findClusters(df, weight):
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
        D = -(Q2 - P2)*(Q3 - P3) - (Q1 - P1)*(Q3-P3) - (Q1 - P1)*(Q2 - P2)
        targetCluster = df.iloc[p]['cluster.ID']
        mask = df['cluster.ID'] == targetCluster
        D[mask] = D[mask]
        D[~mask] = weight*D[~mask]
        A.append(D.tolist())
    delta = 0.1
    X = np.exp(- np.asarray(A) ** 2 / (2. * delta ** 2))
    clustering = SpectralClustering(n_clusters=5,
         assign_labels="discretize", affinity='precomputed',
         random_state=0).fit(X)
    rowlabel = 'new_label_{}'.format(weight)
    print (rowlabel)
    df[rowlabel] = clustering.labels_
    maplabels = {}
    for index, row in df.iterrows():
        el = int(row['cluster.ID'])
        nl = int(row[rowlabel])
        if (maplabels.get(el) == None):
            maplabels[el] = []
            maplabels[el].append([nl, 1])
        else:
            done = 0
            li = maplabels[el]
            for elem in li:
                if (elem[0] == nl):
                    elem[1] = elem[1] + 1
                    done = 1
            if done == 0:
                maplabels[el].append([nl, 1])
    print (maplabels)
    mapto = {}
    for key, value in maplabels.items():
        maxval = 0
        for pair in value:
            if (pair[1] > maxval and mapto.get(pair[0]) == None):
                maxval = pair[1]
                maxarg = pair[0]
        mapto[maxarg] = key
    df[rowlabel].replace(mapto, inplace=True)

def get_cmap(n, name='tab20'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def get_sorted_cid_and_plot(df, fig, ax):
    n_points = df.shape[0]
    cluster_ids_unique = df['new_label'].unique()
    n_clusters = len(cluster_ids_unique) ;
    print('Cluster statistics ...{} points, {} clusters\n'.format(n_points, n_clusters))
#    points = []
#    for cluster_id in cluster_ids_unique:
#        rows = df[df['cluster.ID'] == cluster_id]
#        print(' Cluster ID = {} has {} points'.format(cluster_id, len(rows)))
#        points.append(len(rows))
#    sorted_cid = [x for _,x in sorted(zip(points,cluster_ids_unique))]
    sorted_cid = sorted(cluster_ids_unique)
    ## ---- Find colors for clusters
    #cols = get_cmap(len(cluster_ids_unique))
    cols = get_cmap(19)
    for cluster_id in sorted_cid:
        rows = df[df['new_label'] == cluster_id]
        ax.scatter(rows['x'], rows['y'], color=cols(cluster_id), label=str(cluster_id))
    plt.legend(loc='right', title='Clusters')
    plt.xlabel('x')
    plt.ylabel('y')

def channel_flow(df):
    fig, ax = plt.subplots()
    get_sorted_cid_and_plot(df, fig, ax)
    plt.yscale('log')
    #plt.show()

def plot_embedding(df):
    reducer = umap.UMAP(random_state=42)
    scaled_data = StandardScaler().fit_transform(df)
    embedding = reducer.fit_transform(scaled_data)
    print (embedding.shape)

    for i in range(2, 90):
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        plt.subplot(1, 2, 1)
        ax = fig.gca()
        ax.set_aspect('equal')
        ax.set_title('Original, using GMM')
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            s = 10,
            c=[sns.color_palette()[x] for x in df['cluster.ID']])
        plt.subplot(1, 2, 2)
        ax = fig.gca()
        ax.set_aspect('equal')
        ax.set_title('Using Spectral Embedding')
        weight = 0.5*i
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            s = 10,
            c=[sns.color_palette()[x] for x in df['new_label_{}'.format(weight)]])
        plt.savefig('./figures/new_label_{}.png'.format(weight))

def pca_analysis(df):
    scaled_data = StandardScaler().fit_transform(df)
    pca = PCA(n_components=19)
    pca.fit(scaled_data)
    print (pca.explained_variance_ratio_, pca.n_components)


def main():
    df = pd.read_csv('../toUFl.20210222/channel.flow/data/channel_clustering_11-4-20.dat', header=0, names=['C_1', 'C_2', 'C_3', 'eta_1', 'lambda_1', 'eta_4', 'eta_3', 'case.no', 'cluster.ID', 'x', 'y', 'z'])
    for i in range(2, 90):
        weight = 0.5*i
        findClusters(df, weight)
    plot_embedding(df)
    cdf = pd.read_csv('../toUFl.20210301/allChannel_features.dat', header=0, names=['y_plus', 'xB', 'yB', 'C_1', 'C_2', 'C_3', 'cos_theta', 'phi', 'zeta', 'lambda_1', 'lambda_5', 'eta_1', 'eta_2', 'eta_3', 'eta_4', 'eta_5', 'P_epsilon', 'velocity_y', 'case.no'], delim_whitespace=True)
    #pca_analysis(cdf)
    #channel_flow(df)

if __name__=="__main__":
    main()

