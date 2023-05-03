from sklearn import preprocessing
import numpy as np

# Training data Zero-centered 
def normalize_mean(X):
    """
    Using scikit learn preprocessing to transform feature matrix
    using StandardScaler with mean and standard deviation
    """
    ### BEGIN YOUR CODE (3 points)
    scaler = preprocessing.StandardScaler(with_std=False)
    X = scaler.fit_transform(X)
    ### END YOUR CODE
    return X, scaler.mean_

# Test data Zero-centered
def apply_normalize_mean(X, scaler_mean):
    """
    Apply normalizaton to a testing dataset that have been fit using training dataset.
    
    @arguments:
    X: #frames, #features (in case we use mfcc, #features is 39)
    scaler_mean: mean of fitted StandardScaler that you used in normalize_mean function.
    
    @returns:
    X: normalized matrix
    """ 
    ### BEGIN YOUR CODE (2 points)
    X = X - scaler_mean
    # X = X / scaler_std
    ### END YOUR CODE
    return X
