import numpy as np

################################################
#         Activations
################################################
class Sigmoid():
    def __init__(self, c=1, b=0):
        self.c = c
        self.b = b

    def value(self, x):
        val = 1 + np.exp(-self.c*(x + self.b))
        return 1/val

    def diff(self, x):
        y = self.value(x)
        val = self.c*y*(1-y)
        return val

class Tanh():
    def __init__(self):
        pass

    def value(self, x):
        num = np.exp(x) - np.exp(-x)
        denom = np.exp(x) + np.exp(-x)
        return num/denom

    def diff(self, x):
        y = self.value(x)
        val = 1 - y**2
        return val

class Relu():
    def __init__(self):
        pass

    def value(self, x):
        val = x
        val[val<0] = 0
        return val

    def diff(self, x):
        val = np.ones(x.shape)
        val[val<=0] = 0
        return val

class Softmax():
    def __init__(self):
        pass

    def value(self, x):
        val = np.exp(x)/np.sum(np.exp(x))
        return val

    def diff(self, x):
        y = self.value(x)
        val = (np.ones(y.shape) - y.T)*y
        return val