import numpy as np
from activations import Sigmoid, Tanh, Relu, Softmax

map_activations = {"Sigmoid":Sigmoid(), "Tanh":Tanh(), "Relu":Relu(), "Softmax":Softmax()}

################################################
#         Layers
################################################
class Input():
    def __init__(self, data):
        self.name = "Input"
        self.input = data
        self.size = self.input.shape[0]
        # self.input = np.append(data, np.ones((1, data.shape[1])), axis=0)
        # Having the input as the activated output 
        # to be given to the next layer
        self.a = self.input
        self.type = "Input layer"

    def __repr__(self):
        representation = self.type + " - of Size:" + str(self.size)
        return representation

class Dense():
    def __init__(self, size, activation, name, last=False):
        self.name = name
        self.size = size
        self.activation = map_activations[activation]
        self.activation_name = activation
        self.type = "Dense layer"

    def __repr__(self):
        representation = self.type + " - of Size:" + str(self.size) + "; Activation:" + self.activation_name
        return representation