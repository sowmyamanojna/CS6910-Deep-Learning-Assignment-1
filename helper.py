import numpy as np

################################################
#         Additional Helper Fucntions
################################################
class OneHotEncoder():
    def __init__(self):
        pass

    def fit_transform(self, y, num_classes):
        transformed = np.zeros((num_classes, y.size))
        for col,row in enumerate(y):
            transformed[row, col] = 1

        return transformed


class MinMaxScaler():
    def __init__(self):
        pass

    def fit_transform(self, X):
        transformed = (X - np.min(X, axis=0))/(np.max(X, axis=0)-np.min(X, axis=0))
        return transformed