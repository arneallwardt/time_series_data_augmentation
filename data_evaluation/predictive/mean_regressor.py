import numpy as np
import torch
class MeanRegressor():

    def __init__(self):
        self.rng = np.random.default_rng(42)

    def make_predictions(self, X, returns=False):

        '''
        Make predictions by sampling from a normal distribution with mean and standard deviation calculated from the input data

        Args:  
            - X: np.array of shape (no, seq_len, dim) containing the input data

        Output:
            - y_pred: np.array of shape (no, seq_len) containing the mean of the target variable
        '''

        if not isinstance(X, np.ndarray):
            X = X.numpy()
        
        if returns:
            X_diff = X[:, :, 0]
        else:
            X_diff = np.diff(X[:, :, 0], axis=1) # Calculate the difference between consecutive close prices

        # get mean and standard deviation of the differences for creating the normal distribution later on
        X_mean = np.mean(X_diff, axis=1)
        X_std = np.std(X_diff, axis=1)

        y_pred_diff = self.rng.normal(X_mean, X_std) # sample from the normal distribution
        y_pred = np.array(y_pred_diff) + X[:, -1, 0]

        return y_pred

