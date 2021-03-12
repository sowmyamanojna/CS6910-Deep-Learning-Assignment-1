import math
import numpy as np

################################################
#         Network
################################################

class NeuralNetwork():
    def __init__(self, layers, loss, batch_size, intialization, epochs):
        self.layers = layers
        self.batch_size = batch_size
        self.loss = loss
        self.intialization = intialization
        self.epochs = epochs
        self.num_batches = math.ceil(layers[0].size/batch_size)
        self.param_init()

    def param_init(self):
        size_prev = layers[0].size
        for layer in self.layers[1:]:
            layer.W_size = (size_prev, layer.size)
            size_prev = layer.size

        if self.intialization == "RandomNormal":
            for layer in self.layers[1:]:
                self.W = np.random.normal(loc=0, scale=1.0, size = layer.W_size)
                self.b = np.random.normal(loc=0, scale=1.0, size = (layer.W_size[0],1))

        elif self.intialization == "XavierUniform":
            for layer in self.layers[1:]:
                upper_bound = np.sqrt(6.0/(np.sum(layer.W_size)))
                lower_bound = -1*upper_bound
                self.W = np.random.uniform(low=lower_bound, high=upper_bound, size=layer.W_size)
                self.b = np.zeros((layer.W_size[0],1), dtype = np.float64)


    def forward_propogation(self, epochs):
        X = self.layers[0].input
        for i in range(1, len(self.layers)):
            # Pre-activation
            self.layers[i].h = self.layers[i].W @ self.layers[i-1].a + self.layers[i].b
            # Activation
            self.layers[i].a = self.layers[i].activation.value(self.layers[i].h)

    def backward_propogation(self, t):
        for epoch in range(self.epochs):
            for i in range(self.num_batches):
                X_batch, y_batch = self.layers[0].input[i*self.batch_size:(i+1)*self.batch_size], t[i*self.batch_size:(i+1)*self.batch_size]
                self.layers[-1].loss = 
                for layer in self.layers.reverse():
