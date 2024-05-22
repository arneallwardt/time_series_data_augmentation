import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform 

class DataComparator:

    def maximum_mean_discrepancy(X, Y):
        '''
        maximum mean discrepancy (mmd) is a metric for comparing the distribution of 2 datasets
        '''

        def _gaussian_kernel(x, y, sigma=1.0):
            return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))
        
        XX = squareform(pdist(X, metric=lambda x, y: _gaussian_kernel(x, y)))
        YY = squareform(pdist(Y, metric=lambda x, y: _gaussian_kernel(x, y)))
        XY = np.array([[_gaussian_kernel(x, y) for y in Y] for x in X])
        return XX.mean() + YY.mean() - 2 * XY.mean()
