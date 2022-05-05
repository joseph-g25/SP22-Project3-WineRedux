import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_openml

from utils.data_handling_lib import split_dataset, RANDOM_STATE

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, cache=True)

train, train_labels, test, test_labels = split_dataset(X,y)

def create_pipeline():
    pipeline = Pipeline([
        ("kmeans", KMeans(n_clusters=25, random_state=RANDOM_STATE)),
        ("log_reg", LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=RANDOM_STATE))
    ])
    
    return pipeline