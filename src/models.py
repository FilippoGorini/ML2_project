"""Models and their hyperparameter grids: name -> (constructor, grid searched on validation).

All models predict the 4 legs at once (4 independent binary outputs).
"""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = {
    # ignores the input, always predicts each leg's most frequent training label
    "majority": (lambda: DummyClassifier(strategy="most_frequent"), {}),

    # least squares on targets -1/+1 with an L2 penalty alpha*||w||^2, predicts contact if score > 0.
    # Inputs standardized first so the penalty treats every channel the same, whatever its unit.
    "ridge": (lambda: make_pipeline(StandardScaler(), RidgeClassifier()),
              {"ridgeclassifier__alpha": np.logspace(-3, 5, 9)}),
}
