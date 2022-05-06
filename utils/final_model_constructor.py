"""
Project-specific functions to create and process the final and comparison model objects.
"""
# Top-level utility imports
#import pandas as pd
#import numpy as np

# Top-level sklearn model imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler

from utils.data_handling_lib import RANDOM_STATE

def make_final_model(X, y):
    """
    Creates and fits the final model to the training data.

    Included pipelines:
        Clustering pipeline

        Number pipeline

        Data preparation pipeline

        Model construction pipeline

    Args:
        X (iterable): Wine training features dataset
        y (iterable): Wine training labels dataset

    Returns:
        Pipeline: Final fitted pipeline object
    """

    cluster_features = ["free sulfur dioxide", "total sulfur dioxide", "pH", "sulphates"]
    scale_features = ["fixed acidity", "volatile acidity", "density",
                      "residual sugar", "chlorides", "citric acid"]

    cluster_attribs = X.columns(cluster_features)
    scale_attribs = X.columns(scale_features)
    cat_attribs = X.columns(["color"])

    cluster_pipeline = Pipeline([
        ("kmeans cluster", KMeans(n_clusters=3, random_state=RANDOM_STATE))
        ])

    num_pipeline = Pipeline([
        ("std_scaler", StandardScaler()),
        ])

    data_preparation_pipeline = ColumnTransformer([
        ("cluster", cluster_pipeline, cluster_attribs),
        ("num", num_pipeline, scale_attribs),
        ("pass", "passthrough", cat_attribs)
        ])

    model_pipeline = Pipeline([
        ("preprocessing", data_preparation_pipeline),
        ("rf_reg", RandomForestRegressor(random_state=RANDOM_STATE)),
        ])

    model_pipeline.fit(X=X, y=y)

    return model_pipeline

def make_comparison_model(X, y):
    """
    Creates and fits the comparison model to the training data.

    Included pipelines:
        Number pipeline
        
        Data preparation pipeline

        Model construction pipeline

    Args:
        X (iterable): Wine training features dataset
        y (iterable): Wine training labels dataset

    Returns:
        Pipeline: Final fitted pipeline object
    """

    scale_features = ["fixed acidity", "volatile acidity", "density",
                      "residual sugar", "chlorides", "citric acid"]

    scale_attribs = X.columns(scale_features)
    cat_attribs = X.columns(["color"])

    num_pipeline = Pipeline([
        (),
        ])

    data_preparation_pipeline = ColumnTransformer([
        ("num", num_pipeline, scale_attribs),
        ("pass", "passthrough", cat_attribs)
        ])

    model_pipeline = Pipeline([
        ("preprocessing", data_preparation_pipeline),
        ("rf_reg", RandomForestRegressor(random_state=RANDOM_STATE)),
        ])

    model_pipeline.fit(X=X, y=y)

    return model_pipeline
