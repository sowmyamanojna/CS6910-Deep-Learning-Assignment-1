import numpy as np

################################################
#         Layers
################################################
class Input():
    def __init__(self, data):
        self.name = "Input"
        self.input = data.reshape(-1,1)
        self.input = np.append(self.input, 1).reshape(-1,1)
        # Having the input as the activated output 
        # to be given to the next layer
        self.a = self.input
        self.size = self.input.size

class Dense():
    def __init__(self, size, activation, intialization, name):
        self.name = name
        self.size = size
        self.activation = activation