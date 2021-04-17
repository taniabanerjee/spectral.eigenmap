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
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm
from sklearn import metrics
import itertools

#import plotly.express as px

def get_cartesian_from_barycentric(b):
    t = np.transpose(np.array([[0,0],[1,0],[0.5,sqrt(3)/2]])) # Triangle
    return t.dot(b)

def channel_flow(df):
    fig, ax = plt.subplots()
    get_sorted_cid_and_plot(df, fig, ax)
    plt.yscale('log')
    #plt.show()

def plot_bic(X):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 10)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    
    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])
    clf = best_gmm
    bars = []
    
    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)

    return 

def plot_gmm_clusters(components, df, comp):
    plt.show(block=False)
    labels_pred = GaussianMixture(n_components=comp).fit_predict(components)
    df['gmm-pca'] = labels_pred
    for i in range (0, comp):
        y = df[df['gmm-pca'] == i]
        yp = y['y_plus']
        print ('Cluster', i, min(yp), max(yp), yp.size)
    fig = plt.figure()
    ax = fig.gca()
    cx = []
    cy = []
    for i in range (0, comp):
        y = df[df['gmm-pca'] == i]
        yp = y['y_plus']
        yy = [[i] * yp.size]
        cx = cx + list(yp)
        cy = cy + yy[0]
        index = int(yp.size/10)
        plt.text(yp.iloc[index]+1, yy[0][index]+0.2, '{}, {}, {}'.format(round(min(yp), 2), round(max(yp), 2), yp.size))
    scatter = ax.scatter(cx, cy, c=cy, cmap='tab10', s=5)
    plt.xscale('log')
    plt.xlim([1, 5000])
    plt.ylim([-1, comp])
    plt.xlabel('$y^{+}$')
    plt.ylabel('Cluster')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    plt.scatter(components[:, 0], components[:, 1], c=df['gmm-pca'], cmap='tab10')
    plt.xlabel('1st principal component')
    plt.ylabel('2nd principal component')
    plt.show()

def plot_orig_clusters(df):
    for i in range (1, 6):
        y = df[df['cluster.ID'] == i]
        yp = y['y_plus']
        print ('Cluster', i, min(yp), max(yp), yp.size)
    fig = plt.figure()
    ax = fig.gca()
    cx = []
    cy = []
    for i in range (1, 6):
        y = df[df['cluster.ID'] == i]
        yp = y['y_plus']
        yy = [[i] * yp.size]
        cx = cx + list(yp)
        cy = cy + yy[0]
        index = int(yp.size/10)
        plt.text(yp.iloc[index]+1, yy[0][index]+0.2, '{}, {}, {}'.format(round(min(yp), 2), round(max(yp), 2), yp.size))
    scatter = ax.scatter(cx, cy, c=cy, cmap='tab10', s=5)
    plt.xscale('log')
    plt.xlim([1, 5000])
    plt.ylim([0, 6])
    plt.xlabel('$y^{+}$')
    plt.ylabel('Cluster')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()

def pca_analysis_nd(scaled_data, df, n):
    pca = PCA(n_components=n)
    components = pca.fit_transform(scaled_data)
    #plot_bic(components)
    for j in range(1, 10):
        gmPDF = GaussianMixture(n_components=j, random_state=0, covariance_type='full').fit(components)
        # find clustering performance
        labels_true = df['cluster.ID']
        labels_pred = GaussianMixture(n_components=j, random_state=0).fit_predict(components)
        print ('components', j)
        #print (metrics.rand_score(labels_true, labels_pred))
        print (metrics.adjusted_rand_score(labels_true, labels_pred))
        print (metrics.adjusted_mutual_info_score(labels_true, labels_pred))
        print (metrics.homogeneity_score(labels_true, labels_pred))
        print (metrics.completeness_score(labels_true, labels_pred))
        print (metrics.fowlkes_mallows_score(labels_true, labels_pred))

    plot_gmm_clusters(components, df, 4)
    plot_gmm_clusters(components, df, 5)
    plot_orig_clusters(df)

#    legend = ax.legend(*scatter.legend_elements(), title='Cluster')
#    plt.xlim([1, 5000])
#    plt.xscale('log')
#    plt.xlabel('$y^{+}$')
#    plt.ylabel('$U^{+}$')
#    plt.grid(b=True, which='major', color='#666666', linestyle='-')
#    plt.minorticks_on()
#    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#    plt.show()
#    print (pca.explained_variance_ratio_, pca.n_components)
#    fig = plt.figure()
#    ax = fig.gca()
#    scatter = ax.scatter(df['y_plus'], df['velocity_y'], c=labels_pred, cmap='tab10')
#    legend = ax.legend(*scatter.legend_elements(), title='Cluster')
#    plt.xlim([1, 5000])
#    plt.xscale('log')
#    plt.xlabel('$y^{+}$')
#    plt.ylabel('$U^{+}$')
#    plt.grid(b=True, which='major', color='#666666', linestyle='-')
#    plt.minorticks_on()
#    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#    plt.show()
#    print (pca.explained_variance_ratio_, pca.n_components)

def plot_3d(scaled_data, df):
    pca = PCA(n_components=3)
    components = pca.fit_transform(scaled_data)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.scatter(components[:, 0], components[:, 1], components[:, 2], c=df['cluster.ID'], cmap='tab10')
    ax.set_xlim(-3, 20)
    ax.set_ylim(-3, 20)
    #ax.set_zlim(-2, 2)
    ax.set_xlabel('1st principal component')
    ax.set_ylabel('2nd principal component')
    ax.set_zlabel('3rd principal component')
    plt.show()

def plot_2d(scaled_data, df):
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_data)
    plt.scatter(components[:, 0], components[:, 1], c=df['cluster.ID'], cmap='tab10')
    plt.xlabel('1st principal component')
    plt.ylabel('2nd principal component')
    plt.show()

def plot_without_labels(scaled_data, metric):
    start = 2
    max_comp = 7
    c = ['black', 'white', 'purple', 'orange', 'yellow', 'green', 'violet', 'blue', 'red', 'brown', 'black']
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    title = 'sample'
    gcomprange = range(2, 10)
    for i in range(start,max_comp):
        pca = PCA(n_components=i)
        components = pca.fit_transform(scaled_data)
        scores = []
        for j in gcomprange:
            gmPDF = GaussianMixture(n_components=j, random_state=0).fit(components)
            # find clustering performance
            labels_pred = GaussianMixture(n_components=j, random_state=0).fit_predict(components)
            if (metric == 1):
                scores.append(metrics.silhouette_score(components, labels_pred, metric='euclidean'))
                title = 'silhouette_score'
            elif (metric == 2):
                scores.append(metrics.calinski_harabasz_score(components, labels_pred))
                title = 'variance_ratio_criterion'
                ax.set_ylim(1000, 4000)
            elif (metric == 3):
                scores.append(metrics.davies_bouldin_score(components, labels_pred))
                title = 'davies_bouldin_score'
                ax.set_ylim(0, 2.5)
        ax.plot(gcomprange, scores, 'o-', color=c[i], label='pca-{}'.format(i))
        print (scores)
    ax.set_xlabel('gmm components')
    ax.set_ylabel('score')
    ax.set_xlim(1, 10)
    ax.legend()
    ax.grid(b=True, which='major', color='#666666', linestyle='-')
    ax.minorticks_on()
    ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.title(title)
    plt.show()

    plt.show()

def plot_adjusted_rand_score(scaled_data, df, metric):
    start = 2
    max_comp = 7
    c = ['black', 'white', 'purple', 'orange', 'yellow', 'green', 'violet', 'blue', 'red', 'brown', 'black']
    fig, ax = plt.subplots()
    title = 'sample'
    gcomprange = range(1, 10)
    for i in range(start,max_comp):
        pca = PCA(n_components=i)
        components = pca.fit_transform(scaled_data)
        scores = []
        for j in gcomprange:
            gmPDF = GaussianMixture(n_components=j, random_state=0).fit(components)
            # find clustering performance
            labels_true = df['cluster.ID']
            labels_pred = GaussianMixture(n_components=j, random_state=0).fit_predict(components)
            if (metric == 1):
                scores.append(metrics.adjusted_rand_score(labels_true, labels_pred))
                title = 'adjusted_rand_score'
            elif (metric == 2):
                scores.append(metrics.adjusted_mutual_info_score(labels_true, labels_pred))
                title = 'adjusted_mutual_info_score'
            elif (metric == 3):
                scores.append(metrics.homogeneity_score(labels_true, labels_pred))
                title = 'homogeneity_score'
            elif (metric == 4):
                scores.append(metrics.completeness_score(labels_true, labels_pred))
                title = 'completeness_score'
            elif (metric == 5):
                scores.append(metrics.fowlkes_mallows_score(labels_true, labels_pred))
                title = 'fowlkes_mallows_score'
        ax.plot(gcomprange, scores, 'o-', color=c[i], label='pca-{}'.format(i))
        print (scores)
    ax.set_xlabel('gmm components')
    ax.set_ylabel('score')
    ax.set_ylim(0, 1)
    ax.set_xlim(1, 10)
    ax.legend()
    ax.grid(b=True, which='major', color='#666666', linestyle='-')
    ax.minorticks_on()
    ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.title(title)
    plt.show()

    plt.show()

def plot_error_reconstruction(scaled_data, df):
    from numpy import linalg as LA
    max_comp=14
    pca = PCA(n_components=max_comp)
    pca2_results = pca.fit_transform(scaled_data)
    print (pca.explained_variance_ratio_, pca.n_components)
    print (pca.explained_variance_)
    print (pca.explained_variance_ratio_)
    print (pca.explained_variance_ratio_.cumsum())
    cumsum_variance = pca.explained_variance_ratio_.cumsum().copy()
    start=1
    error_record=[]
    for i in range(start,max_comp+1):
        pca = PCA(n_components=i)
        pca2_results = pca.fit_transform(scaled_data)
        pca2_proj_back=pca.inverse_transform(pca2_results)
        total_loss=LA.norm((scaled_data-pca2_proj_back),None)
        error_record.append(total_loss)
    
    fig, ax1 = plt.subplots()
    #plt.clf()
    #plt.figure()
    #plt.title("reconstruct error of pca")
    #plt.plot(error_record,'ro-')
    color = 'tab:red'
    ax1.plot(error_record, 'o-', color=color)
    ax1.set_ylabel('error', color=color)
    ax1.set_xlabel('pca components')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('variance', color=color)
    ax2.plot(cumsum_variance,'*-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    #plt.plot(cumsum_variance,'b*-')
    plt.xticks(range(len(error_record)), range(start,max_comp+1), rotation='vertical')
    plt.xlim([-1, len(error_record)])
    ax1.grid(b=True, which='major', color='#666666', linestyle='-')
    ax1.minorticks_on()
    ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()

def pca_analysis_2d(scaled_data, df):
    plot_without_labels(scaled_data, 1)
    plot_without_labels(scaled_data, 2)
    plot_without_labels(scaled_data, 3)
    plot_adjusted_rand_score(scaled_data, df, 1)
    plot_adjusted_rand_score(scaled_data, df, 2)
    plot_adjusted_rand_score(scaled_data, df, 3)
    plot_adjusted_rand_score(scaled_data, df, 4)
    plot_adjusted_rand_score(scaled_data, df, 5)
    #plot_error_reconstruction(scaled_data, df)
    #plot_3d(scaled_data, df)
    plot_2d(scaled_data, df)
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(scaled_data)
    plot_bic(components)
    fig = plt.figure()
    for j in range(1, 10):
        plt.subplot(3, 3, j)
        ax = fig.gca()
        gmPDF = GaussianMixture(n_components=j, random_state=0).fit(components)
        # find clustering performance
        labels_true = df['cluster.ID']
        labels_pred = GaussianMixture(n_components=j, random_state=0).fit_predict(components)
        print ('components', j)
        #print (metrics.rand_score(labels_true, labels_pred))
        print (metrics.adjusted_rand_score(labels_true, labels_pred))
        print (metrics.adjusted_mutual_info_score(labels_true, labels_pred))
        print (metrics.homogeneity_score(labels_true, labels_pred))
        print (metrics.completeness_score(labels_true, labels_pred))
        print (metrics.fowlkes_mallows_score(labels_true, labels_pred))
        # display predicted scores by the model as a contour plot
        x = np.linspace(-10., 20.)
        y = np.linspace(-10., 15.)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        Z = -gmPDF.score_samples(XX)
        Z = Z.reshape(X.shape)
        
        CS = ax.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                         levels=np.logspace(0, 3, 10))
        #CB = ax.colorbar(CS, shrink=0.8, extend='both')

        scatter = ax.scatter(components[:, 0], components[:, 1], c=df['cluster.ID'], cmap='tab10')
        legend = ax.legend(*scatter.legend_elements(), title='Cluster')
        plt.xlabel('1st principal component')
        plt.ylabel('2nd principal component')
    plt.show()
    print (pca.explained_variance_ratio_, pca.n_components)

def main():
    df = pd.read_csv('../toUFl.20210222/channel.flow/data/channel_clustering_11-4-20.dat', header=0, names=['C_1', 'C_2', 'C_3', 'eta_1', 'lambda_1', 'eta_4', 'eta_3', 'case.no', 'cluster.ID', 'x', 'y', 'z'])
    cdf = pd.read_csv('../toUFl.20210301/allChannel_features.dat', header=None, names=['y_plus', 'xB', 'yB', 'C_1', 'C_2', 'C_3', 'cos_theta', 'phi', 'zeta', 'lambda_1', 'lambda_5', 'eta_1', 'eta_2', 'eta_3', 'eta_4', 'eta_5', 'P_epsilon', 'velocity_y', 'case.no'], delim_whitespace=True)
    cdf['cluster.ID'] = df['cluster.ID']
#    columns = cdf.columns
#    for column in columns:
#        if (column == 'y_plus' or column == 'velocity_y' or column == 'case.no'):
#            continue
#        else:
#            data_mean, data_std = mean(cdf[column]), std(cdf[column])
#            cut_off = data_std * 8
#            lower, upper = data_mean - cut_off, data_mean + cut_off
#            outliers = []

    relevantdf = pd.DataFrame()
    #relevantdf['y_plus'] = cdf['y_plus']
    #relevantdf['C_1'] = cdf['C_1']
    #relevantdf['C_2'] = cdf['C_2']
    #relevantdf['C_3'] = cdf['C_3']
    relevantdf['xB'] = cdf['xB']
    relevantdf['yB'] = cdf['yB']
    relevantdf['cos_theta'] = cdf['cos_theta']
    relevantdf['phi'] = cdf['phi']
    relevantdf['zeta'] = cdf['zeta']
    relevantdf['lambda_1'] = cdf['lambda_1']
    relevantdf['lambda_5'] = cdf['lambda_5']
    relevantdf['eta_1'] = cdf['eta_1']
    relevantdf['eta_2'] = cdf['eta_2']
    relevantdf['eta_3'] = cdf['eta_3']
    relevantdf['eta_4'] = cdf['eta_4']
    relevantdf['eta_5'] = cdf['eta_5']
    relevantdf['P_epsilon'] = cdf['P_epsilon']
    #relevantdf['velocity_y'] = cdf['velocity_y']
    scaled_data = StandardScaler().fit_transform(relevantdf)
    #pca_analysis_2d(scaled_data, df)
    #pca_analysis_nd(scaled_data, cdf, 2)
    pca_analysis_nd(scaled_data, cdf, 3)
    #pca_analysis_nd(scaled_data, cdf, 4)
    rdf = pd.DataFrame()
    rdf['C_1'] = df['C_1']
    rdf['C_2'] = df['C_2']
    rdf['C_3'] = df['C_3']
    rdf['eta_1'] = df['eta_1']
    rdf['lambda_1'] = df['lambda_1']
    rdf['eta_4'] = df['eta_4']
    rdf['eta_3'] = df['eta_3']
    scaled_reduced_data = StandardScaler().fit_transform(rdf)
    #pca_analysis_2d(scaled_reduced_data, df)
    #pca_analysis_nd(scaled_reduced_data, cdf, 3)
    labels_true = df['cluster.ID']
    labels_pred = GaussianMixture(n_components=5, random_state=0).fit_predict(scaled_data)
    print ('components', 5)
    #print (metrics.rand_score(labels_true, labels_pred))
    print (metrics.adjusted_rand_score(labels_true, labels_pred))
    print (metrics.adjusted_mutual_info_score(labels_true, labels_pred))
    print (metrics.homogeneity_score(labels_true, labels_pred))
    print (metrics.completeness_score(labels_true, labels_pred))
    print (metrics.fowlkes_mallows_score(labels_true, labels_pred))

    plt.scatter(cdf['y_plus'], cdf['velocity_y'], c=df['cluster.ID'], cmap='tab10', s=0.5)
    plt.xlim([1, 5000])
    plt.xscale('log')
    plt.xlabel('$y^{+}$')
    plt.ylabel('$U^{+}$')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()

    labels_pred = GaussianMixture(n_components=5, random_state=0, covariance_type='full').fit_predict(scaled_reduced_data)
    plt.scatter(cdf['y_plus'], cdf['velocity_y'], c=labels_pred, cmap='tab10', s=0.5)
    plt.xlim([1, 5000])
    plt.xscale('log')
    plt.xlabel('$y^{+}$')
    plt.ylabel('$U^{+}$')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()
    #channel_flow(df)

if __name__=="__main__":
    main()

