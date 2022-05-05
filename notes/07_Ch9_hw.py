"""
1. What is unsupervised learning?
    Unsupervised learning describes a type of machine learning system where 
    the data is unlabeled, meaning that it does not have the associated
    desired solutions to compare against.

2. Describe what the \(K\)-means clustering algorithm seeks to do when we train it on a dataset.
    words

3. Now describe how the \(K\)-means clustering algorithm works -- what are the steps in discovering the cluster?
    words

4. Go find that script we wrote that generated two disctincts blobs of data. Write a new sscript that uses the same idea to generate 3-4 disctinct blobs of points
in 2 dimensions, and run K-means on it. Do what you need to do to see if the algorithm found the right clusters.
"""

from re import X
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

RANDOM_STATE = int(19)

def split_dataset(data, labels):
    train_set, test_set, train_set_labels, test_set_labels = train_test_split(data, labels, test_size=0.3, random_state=RANDOM_STATE)
    
    return train_set, test_set, train_set_labels, test_set_labels

X, y = make_moons(n_samples=10000, noise=None, random_state=RANDOM_STATE)

#train_set, test_set, train_set_labels, test_set_labels = split_dataset(X, y)

plt.scatter(x=X, y=X)

plt.show()