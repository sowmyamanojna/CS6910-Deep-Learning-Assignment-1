import numpy as np

################################################
#         Additional Helper Fucntions
################################################
class OneHotEncoder():
    def __init__(self):
        pass
    
    def fit(self, y, num_classes):
        self.y = y
        self.num_classes = num_classes

    def transform(self):
        transformed = np.zeros((self.num_classes, self.y.size))
        for col,row in enumerate(self.y):
            transformed[row, col] = 1
        return transformed

    def fit_transform(self, y, num_classes):
        self.fit(y, num_classes)
        return self.transform()

    def inverse_transform(self, y):
        # Assumes direct correation between the position and class number
        y_class = np.argmax(y, axis=0)
        return y_class


class MinMaxScaler():
    def __init__(self):
        pass

    def fit(self, X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)

    def transform(self, X):
        transformed = (X - self.min)/(self.max-self.min)
        return transformed

    def fit_transform(self, X):
        self.fit(X)
        self.transform(X)