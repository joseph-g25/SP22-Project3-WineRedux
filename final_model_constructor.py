# Top-level utility imports
import pandas as pd
import numpy as np

# Top-level sklearn model imports
from sklearn.cluster import KMeans

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Local utility imports
import utils.data_handling_lib as dhl

from utils.data_handling_lib import RANDOM_STATE

def create_pipeline(X, y):
    """
    AI is creating summary for create_pipeline

    Args:
        X ([type]): [description]
        y ([type]): [description]

    Returns:
        [type]: [description]
    """
    data_preparation_pipeline = Pipeline([
        ("kmeans_cluster", KMeans(random_state=RANDOM_STATE)),
        ("ColumnTransformer", ColumnTransformer(transformers=idk lmfao))
    ])
    
    model_pipeline = Pipeline([
        #("reg", RegressionModel()),
    ])
    
    model_pipeline.fit(X=X, y=y)
    
    return model_pipeline

