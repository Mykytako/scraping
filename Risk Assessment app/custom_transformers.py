import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CosineTransformer(BaseEstimator, TransformerMixin):
    #Class Constructor
    def __init__(self, feature_names1,feature_names2):
        self.feature_names1 = feature_names1
        self.feature_names2 = feature_names2
        
    #Return self nothing else to do here    
    def fit(self, X, y = None):
        return self 
    
    #Method that describes what we need this transformer to do
    #which to conver hour to cosine
    def transform(self, X, y = None):
        X_ = X.copy()
        X_[self.feature_names1] = np.cos(X_[self.feature_names1] * (2. * np.pi / 24))
        X_[self.feature_names2] = np.cos(X_[self.feature_names2] * (2. * np.pi / 60))
        return X_
    
    #Need to override this method to ensure can use afterwards to extract all feature names
    def get_feature_names_out(self, input_features=None):
        return [self.feature_names1, self.feature_names2]