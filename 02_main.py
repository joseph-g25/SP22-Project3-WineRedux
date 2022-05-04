"""
Main scripting environment to express the data preparation process and pipeline structure. Exports
model upon successful execution.
"""

# Top-level utility imports
import pandas as pd
import numpy as np

# Top-level sklearn model imports
from sklearn.cluster import KMeans

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Local utility imports
import utils.data_handling_lib as dhl
import final_model_constructor as fmc
from utils.data_handling_lib import RANDOM_STATE

raw_dataset = dhl.load_data()

X_train, y_train, X_test, y_test = dhl.strat_split_dataset(X=raw_dataset, label_id="quality")

final_model = fmc.create_pipeline(X=X_train, y=y_train)

dhl.save_model(final_model, "final_model")
