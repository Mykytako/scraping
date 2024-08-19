
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class CosineTransform(BaseEstimator, TransformerMixin):
    def __init__(self, month):
        self.month=month
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.month] = np.cos(X[self.month] * (2 * np.pi / 12))
        return X
