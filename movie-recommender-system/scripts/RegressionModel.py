### Clean-room implementation based on scikit-learn's RidgeRegression class
### Reference: https://github.com/scikit-learn/scikit-learn/blob/6e9039160/sklearn/linear_model/_ridge.py#L1016
### License: BSD 3-Clause License
### Authors: Tyler Ho, Quynh Nguyen
### CS439 - Final Project
### Nov 2024

import numpy as np

class RegressionModel:
    """
    fits regression model and applies to matrix for predictions
    """

    def __init__(self, alpha):
        """
        :param alpha: regularization strength
        """

        self.alpha = alpha
        self.coef = None

    def fit(self, X, y):
        """
        fit model to training data

        :param X: feature matrix
        :param y: target matrix
        """

        n_features = X.shape[1]
        
        #identity matrix to regularize features
        I = np.eye(n_features)

        # calculates regression weights using normal formula
        self.coef = np.linalg.inv(X.T @ X + self.alpha * I) @ X.T @ y

    def predict(self, X):
        """
        makes prediction 

        :param X: feature matrix
        :return: prediction vector
        """

        return X @ self.coef
    


