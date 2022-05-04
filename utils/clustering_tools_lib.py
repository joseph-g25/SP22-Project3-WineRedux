
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

from .data_handling_lib import RANDOM_STATE


def KMeans_cluster_load(data, n_clusters=8, n_init=10, max_iter=300, random_state=RANDOM_STATE):
    kmeans_clf = KMeans(n_clusters=n_clusters,
                        n_init=n_init,
                        max_iter=max_iter,
                        random_state=random_state)

    #kmeans_clf.fit(data)
    return kmeans_clf

def mini_KMeans_cluster_load(X, n_clusters=8, n_init=10,
                             max_iter=300, random_state=RANDOM_STATE):
    mini_kmeans_clf = MiniBatchKMeans()
    
    return mini_kmeans_clf

def give_kmeans_set(data, range_lower=1, range_upper=10, step=1):
    kmeans_set = [KMeans(n_clusters=k, random_state=42).fit(data)
                for k in range(range_lower, range_upper, step)]
    return kmeans_set

def give_inertias_(kmeans_set):
    inertias_ = [model.inertia_ for model in kmeans_set]
    return inertias_

def give_silhouette_scores(data, range_lower=1, range_upper=10, step=1, kmeans_set=None):
    """
    If a set of KMeans classifiers is given, the function will use that to determine silhouette
    scores. Else, a set of KMeans classifiers will be generated according to defined upper and lower
    bounds.
    """
    if kmeans_set is not None:
        silhouette_scores = [silhouette_score(data, model.labels_)
                     for model in kmeans_set[1:]]

        return silhouette_scores

    else:
        kmeans_set = give_kmeans_set(data=data, range_lower=range_lower,
                                    range_upper=range_upper, step=step)

        silhouette_scores = [silhouette_score(data, model.labels_)
                     for model in kmeans_set[1:]]

        return silhouette_scores
