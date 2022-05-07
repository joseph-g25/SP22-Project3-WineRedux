"""
Personal library to streamline the writing and use of matplotlib's pyplot tools.
"""
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
#import pandas as pd

from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=1000):
    """
    Saves the open pyplot figure to exports/images/.

    Args:
        fig_id (str): Filename and identifier of Pyplot plot to write.
        tight_layout (bool, optional): Enables or disables tight layout. Defaults to True.
        fig_extension (str, optional): Sets the image format type. Defaults to "png".
        resolution (int, optional): Desired resolution of the final image in DPI. Defaults to 1000.
    """
    IMAGES_PATH = "exports/images/"
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution, facecolor='white', transparent=False)

def plot_fig(x_dim=8, y_dim=4):
    plt.figure(figsize=(x_dim, y_dim))

def plot_silhoutte_diagram(X, kmeans_set, silhouette_scores, savefig=False):
    """
    AI is creating summary for plot_silhoutte_diagram

    Args:
        X ([type]): [description]
        kmeans_set ([type]): [description]
        silhouette_scores ([type]): [description]
        savefig (bool, optional): Save plot to exports folder. Defaults to False.
    """
    plt.figure(figsize=(11, 9))

    for k in (3, 4, 5, 6):
        plt.subplot(2, 2, k - 2)
        
        y_pred = kmeans_set[k - 1].labels_
        silhouette_coefficients = silhouette_samples(X, y_pred)

        padding = len(X) // 30
        pos = padding
        ticks = []
        for i in range(k):
            coeffs = silhouette_coefficients[y_pred == i]
            coeffs.sort()

            color = mpl.cm.Spectral(i / k)
            plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                            facecolor=color, edgecolor=color, alpha=0.7)
            ticks.append(pos + len(coeffs) // 2)
            pos += len(coeffs) + padding

        plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
        plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
        if k in (3, 5):
            plt.ylabel("Cluster")
        
        if k in (5, 6):
            plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.xlabel("Silhouette Coefficient")
        else:
            plt.tick_params(labelbottom=False)

        plt.axvline(x=silhouette_scores[k - 2], color="red", linestyle="--")
        plt.title("$k={}$".format(k), fontsize=16)

    if savefig is True:
        fig_id_count = 0
        dir_path = r"exports/images/"
        for filepath in os.scandir(dir_path):
            if filepath.is_file():
                fig_id_count += 1
        fig_id = str(fig_id_count) + "_silhoutte_diagram"
        save_fig(fig_id=fig_id)
        plt.show()
    else:
        plt.show()

def plot_silhouette_scores(silhouette_scores, var_name="", savefig=False):
    """
    Generates and shows a graph of scores for each k value of the n_clusters parameter for KMeans clustering.

    Args:
        silhouette_scores (List): A list of silhouette scores for a set of KMeans clustering algorithms.
        var_name (str, optional): String to label the y-axis according to input variable. Defaults to "".
        savefig (bool, optional): Save plot to exports folder. Defaults to False.
    """
    upper = len(silhouette_scores)
    
    plt.figure(figsize=(8, 3))
    plt.plot(range(2, (upper+2)), silhouette_scores, "bo-")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Silhouette score", fontsize=14)
    plt.axis([1, upper+2, 0, 1.2])
    if var_name != "":
        plt.title(label="Set: " + var_name,fontsize=16)
    
    if savefig is True:
        fig_id_count = 0
        dir_path = r"exports/images/"
        for filepath in os.scandir(dir_path):
            if filepath.is_file():
                fig_id_count += 1

        fig_id = str(fig_id_count) + "_silhouette_scores"
        
        save_fig(fig_id=fig_id)
        plt.show()
    else:
        plt.show()

def plot_corr_matrix_heatmap(corr_matrix, dataset, savefig=False):
    """
    Generates and shows a grid of heatmaps for correlations between any 2 variables in a matrix.  

    Args:
        corr_matrix (DataFrame): A correlation matrix of all relevant features to each other feature.
        dataset (DataFrame): Dataset with corresponding feature names to plot.
        savefig (bool, optional): Save plot to exports folder. Defaults to False.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr_matrix,cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(dataset.columns),1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(dataset.columns)
    ax.set_yticklabels(dataset.columns)
    
    if savefig is True:
        fig_id_count = 0
        dir_path = r"exports/images/"
        for filepath in os.scandir(dir_path):
            if filepath.is_file():
                fig_id_count += 1
        fig_id = str(fig_id_count) + "_corr_matrix_heatmap"
        save_fig(fig_id=fig_id)
        plt.show()
    else:
        plt.show()

def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    """
    AI is creating summary for plot_centroids

    Args:
        centroids ([type]): [description]
        weights ([type], optional): [description]. Defaults to None.
        circle_color (str, optional): [description]. Defaults to 'w'.
        cross_color (str, optional): [description]. Defaults to 'k'.
    """
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True, savefig=False):
    """
    AI is creating summary for plot_decision_boundaries

    Args:
        clusterer ([type]): [description]
        X ([type]): [description]
        resolution (int, optional): [description]. Defaults to 1000.
        show_centroids (bool, optional): [description]. Defaults to True.
        show_xlabels (bool, optional): [description]. Defaults to True.
        show_ylabels (bool, optional): [description]. Defaults to True.
        savefig (bool, optional): Save plot to exports folder. Defaults to False.
    """
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

    if savefig is True:
        fig_id_count = 0
        dir_path = r"exports/images/"
        for filepath in os.scandir(dir_path):
            if filepath.is_file():
                fig_id_count += 1
        fig_id = str(fig_id_count) + "_decision_boundaries"
        save_fig(fig_id=fig_id)
        plt.show()
    else:
        plt.show()
