### Tyler Ho, Quynh Nguyen
### CS439
### Final Project
### Fall 2024

import numpy as np

class Scaler:
    """
    normalizes matrix fed into it, scaling to unit variance

    based on StandardScaler from scikit
    """

    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, X):
        """
        calculate mean and sd
    
        :param X: input matrix
        """

        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)

        # avoid division by zero by setting zero std to 1
        self.scale[self.scale == 0] = 1

    
    def transform(self, X):
        """
        apply standard scaling to matrix
        
        :param X: input matrix
        :return: scaled matrix
        """

        # apply scaling factor to matrix
        return (X - self.mean) / self.scale

    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        
        :param X: input matrix
        :return: scaled matrix
        """

        # gets mean and sd
        self.fit(X)

        #applies the scaling factor to input matrix
        return self.transform(X)