"""
Scripting environment to build and present the final model parameters
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
        Clustering pipeline - operates on free sulfur dioxide, total sulfur dioxide, pH, sulphates

        Number pipeline - operates on fixed acidity, volatile acidity, density, residual sugar, chlorides, citric acid

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
    cat_features = ["color"]

    cluster_pipeline = Pipeline([
        ("kmeans_cluster", KMeans(n_clusters=2, random_state=RANDOM_STATE, algorithm="full"))
        ])

    num_pipeline = Pipeline([
        ("std_scaler", StandardScaler(with_mean=True)),
        ])

    data_preparation_pipeline = ColumnTransformer([
        ("cluster", cluster_pipeline, cluster_features),
        ("num", num_pipeline, scale_features),
        ("pass", "passthrough", cat_features)
        ], n_jobs=-1)

    model_pipeline = Pipeline([
        ("preprocessing", data_preparation_pipeline),
        ("rf_reg", RandomForestRegressor(n_estimators=1000, random_state=RANDOM_STATE, bootstrap=True, n_jobs=-1)),
        ])

    model_pipeline.fit(X=X,y=y)

    return model_pipeline

def make_comparison_model(X, y):
    """
    Creates and fits the comparison model to the training data.

    Included pipelines:
        Number pipeline - operates on fixed acidity, volatile acidity, density, residual sugar, chlorides, citric acid

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
    cat_features = ["color"]

    num_pipeline = Pipeline([
        ("std_scaler", StandardScaler(with_mean=True)),
        ])

    data_preparation_pipeline = ColumnTransformer([
        ("num", num_pipeline, scale_features),
        ("pass", "passthrough", cat_features)
        ], n_jobs=-1)

    model_pipeline = Pipeline([
        ("preprocessing", data_preparation_pipeline),
        ("rf_reg", RandomForestRegressor(n_estimators=1000, random_state=RANDOM_STATE, bootstrap=True, n_jobs=-1)),
        ])

    model_pipeline.fit(X=X,y=y)

    return model_pipeline
