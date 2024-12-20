### Clean-room implementation based on scikit-learn's TruncatedSVD class
### Reference: https://github.com/scikit-learn/scikit-learn/blob/fa5d7275b/sklearn/decomposition/_truncated_svd.py#L28
### License: BSD 3-Clause License
### Authors: Tyler Ho, Quynh Nguyen
### CS439 - Final Project
### Nov 2024

import numpy as np

class SVD:
    """
    reduces dimensionality of matrix
    """
    def __init__(self):

        # number of users -> number of vectors
        self.n_components = 610
        # V^T
        self.components = None
        # Sigma
        self.singular_values_ = None

    # perform svd on user-item matrix
    def fit_transform(self, X):
        """
        perform svd on user-movie matrix

        :param X: input matrix
        :return: transformed matrix
        """

        X = X.astype(float)

        # set each component of svd
        U, S, VT = np.linalg.svd(X, full_matrices=False)

        # cut down matrix to number of users
        self.components = VT[:self.n_components]
        self.singular_values_ = S[:self.n_components]

        # Transform X by projecting it onto the top n_components
        transformed_X = U[:, :self.n_components] * self.singular_values_

        return transformed_X
    
