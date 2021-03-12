import numpy as np

################################################
#         Initializers
################################################
class RandomNormal():
    def __init__(self, mean = 0.0, stddev = 1.0):
        self.mean = mean
        self.stddev = stddev
    
    def weights_biases(self, n_prev, n_curr):
        W = np.random.normal(loc = self.mean, scale = self.stddev, \
                             size = (n_prev, n_curr))
        b = np.random.normal(loc = self.mean, scale = self.stddev, \
                             size = (n_curr,))
        return W, b
    
class XavierUniform():
    def __init__(self):
        pass
    
    def weights_biases(self, n_prev, n_curr):
        upper_bound = np.sqrt(6.0/(n_prev + n_curr))
        lower_bound = -1*upper_bound
        W = np.random.uniform(low = lower_bound, high = upper_bound, \
                              size = (n_prev, n_curr))
        b = np.zeros((n_curr,), dtype = np.float64)
        return W, b