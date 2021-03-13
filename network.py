import math
import numpy as np

map_optimizers = {"Momentum":Momentum(), "Nesterov":Nesterov(), "AdaGrad":AdaGrad(), "RMSProp":RMSProp(), "Adam":Adam(), "Nadam":Nadam()}
################################################
#         Network
################################################

class NeuralNetwork():
    def __init__(self, layers, batch_size, optimizer, intialization, epochs, t):
        self.layers = layers
        self.batch_size = batch_size
        self.intialization = intialization
        self.epochs = epochs
        self.optimizer = map_optimizers[optimizer]
        self.num_batches = math.ceil(layers[0].size/batch_size)
        self.t = t
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

        # Final sofmax activation
        self.layers[-1].y = Softmax.values(self.layers[-1].a)

    def backward_propogation(self):
        # Initialize variables neesed to keep track of loss
        self.loss_hist = []
        self.loss = CrossEntropy()

        # Perform Backprop
        for epoch in range(self.epochs):
            for i in range(self.num_batches):
                X_batch = self.layers[0].input[i*self.batch_size:(i+1)*self.batch_size]
                t_batch = t[:, i*self.batch_size:(i+1)*self.batch_size]
                y_batch = self.layers[-1].y[:, i*self.batch_size:(i+1)*self.batch_size]

                # Calculate Loss, grad wrt y and softmax for last layer
                self.loss_hist.append(self.loss.calc_loss(t_batch, y_batch))

                self.layers[-1].cross_grad = self.loss.diff()
                self.layers[-1].softmax_grad = Softmax.diff(X_batch)
                self.layers[-1].a_grad = self.layers[-1].cross_grad*self.layers[-1].softmax_grad
                self.layers[-1].h_grad = self.layers[-1].a_grad * self.layers[-1].activation.diff(X_batch)

                self.layers[-1].W_grad = self.layers[-1].h_grad @ self.layers[-2].a.T
                self.layers[-1].b_grad = self.layers[-1].h_grad

                self.layers[-1].W_update = self.layers[-1].optimizer.get_update(self.layers[-1].W_grad)
                self.layers[-1].b_update = self.layers[-1].optimizer.get_update(self.layers[-1].b_grad)

                # Backpropogation for the remaining layers
                for i in range(len(self.layers[:-1]), -1, -1):
                    self.layers[i].a_grad = self.layers[i+1].W.T @ self.layers[i+1].h_grad
                    self.layers[i].h_grad = self.layers[i].a_grad * self.layers[i].activation.diff(X_batch)
                    self.layers[i].W_grad = self.layers[i].h_grad @ self.layers[-2].a.T
                    self.layers[i].b_grad = self.layers[i].h_grad

                    self.layers[i].W_update = self.layers[i].optimizer.get_update(self.layers[i].W_grad)
                    self.layers[i].b_update = self.layers[i].optimizer.get_update(self.layers[i].b_grad)

                # Update the weights
                for layer in layers:
                    layer.W = layer.W - self.W_update
                    layer.b = layer.b - self.b_update
                    