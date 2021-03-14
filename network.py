import math
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from copy import deepcopy
from activations import Sigmoid, Tanh, Relu, Softmax
from layers import Input, Dense
from optimizers import Momentum, Nesterov, AdaGrad, RMSProp, Adam, Nadam
from layers import Input, Dense
from loss import CrossEntropy

map_optimizers = {"Momentum":Momentum(), "Nesterov":Nesterov(), "AdaGrad":AdaGrad(), "RMSProp":RMSProp(), "Adam":Adam(), "Nadam":Nadam()}
################################################
#         Network
################################################

class NeuralNetwork():
    def __init__(self, layers, batch_size, optimizer, intialization, epochs, t, optim_params=None):
        self.layers = layers
        self.batch_size = batch_size
        self.intialization = intialization
        self.epochs = epochs
        self.optimizer = optimizer
        self.num_batches = math.ceil(layers[0].size/batch_size)
        self.t = t
        self.param_init(optimizer, optim_params)

    def param_init(self, optimizer, optim_params):
        size_prev = self.layers[0].size
        for layer in self.layers[1:]:
            layer.W_size = (layer.size, size_prev)
            size_prev = layer.size
            layer.optimizer = deepcopy(map_optimizers[optimizer])
            if optim_params:
                layer.optimizer.set_params(**optim_params)

            # layer.b_optimizer = map_optimizers[self.optimizer]

        if self.intialization == "RandomNormal":
            for layer in self.layers[1:]:
                layer.W = np.random.normal(loc=0, scale=1.0, size = layer.W_size)
                # self.b = np.random.normal(loc=0, scale=1.0, size = (layer.W_size[0],1))

        elif self.intialization == "XavierUniform":
            for layer in self.layers[1:]:
                # upper_bound = np.sqrt(6.0/(np.sum(layer.W_size)))
                # lower_bound = -1*upper_bound
                initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
                layer.W = np.array(initializer(shape=layer.W_size))
                # layer.W = np.random.uniform(low=lower_bound, high=upper_bound, size=layer.W_size)
                # layer.b = np.zeros((layer.W_size[0],1), dtype = np.float64)


    def forward_propogation(self):
        X = self.layers[0].input
        for i in range(1, len(self.layers)):
            # Pre-activation
            self.layers[i].h = self.layers[i].W @ self.layers[i-1].a #+ self.layers[i].b
            # Activation
            self.layers[i].a = self.layers[i].activation.value(self.layers[i].h)

        # Final sofmax activation
        self.layers[-1].y = Softmax().value(self.layers[-1].a)

    def backward_propogation(self):
        # Initialize variables neesed to keep track of loss
        self.loss_hist = []
        self.loss = CrossEntropy()
        flag = 0

        # Perform Backprop
        for epoch in tqdm(range(self.epochs)):
            for batch in range(self.num_batches):
                # print("\n", "="*50)
                # print("Batch:", batch)
                X_batch = self.layers[0].input[batch*self.batch_size:(batch+1)*self.batch_size]
                t_batch = self.t[:, batch*self.batch_size:(batch+1)*self.batch_size]
                y_batch = self.layers[-1].y[:, batch*self.batch_size:(batch+1)*self.batch_size]
                self.y_batch = y_batch

                # Calculate Loss, grad wrt y and softmax for last layer
                self.loss_hist.append(self.loss.calc_loss(self.t, self.layers[-1].y))
                
                try:
                    if self.loss_hist[-1] > self.loss_hist[-2]:
                        print("Early Stopping")
                        flag = 1
                except:
                    pass

                if flag == 1:
                    break

                self.layers[-1].cross_grad = self.loss.diff()
                # self.layers[-1].softmax_grad = Softmax().diff(self.layers[-1].a)
                # print(self.layers[-1].cross_grad.size, self.layers[-1].softmax_grad.size)
                # self.layers[-1].a_grad = self.layers[-1].cross_grad*self.layers[-1].softmax_grad
                self.layers[-1].a_grad = y_batch - t_batch
                self.layers[-1].h_grad = self.layers[-1].a_grad * self.layers[-1].activation.diff(self.layers[-1].h[:, batch*self.batch_size:(batch+1)*self.batch_size])

                self.layers[-1].W_grad = self.layers[-1].h_grad @ self.layers[-2].a[:, batch*self.batch_size:(batch+1)*self.batch_size].T
                # self.layers[-1].b_grad = self.layers[-1].h_grad

                self.layers[-1].W_update = self.layers[-1].optimizer.get_update(self.layers[-1].W_grad)
                # self.layers[-1].b_update = self.layers[-1].b_optimizer.get_update(self.layers[-1].b_grad)

                # print("Last Layer")
                # print("a_grad shape:", self.layers[-1].a_grad.shape)
                # print("h_grad shape:", self.layers[-1].h_grad.shape)
                # print("W_grad shape:", self.layers[-1].W_grad.shape)
                # print("W_update shape:", self.layers[-1].W_update.shape)
                # print("W_shape:", self.layers[-1].W.shape)
                # print(self.layers[-1].W_update.shape != self.layers[-1].W.shape)
                # print()

                assert self.layers[-1].W_update.shape == self.layers[-1].W.shape, "Sizes don't match"


                # Backpropogation for the remaining layers
                for i in range(len(self.layers[:-2]), 0, -1):
                    self.layers[i].a_grad = self.layers[i+1].W.T @ self.layers[i+1].h_grad
                    self.layers[i].h_grad = self.layers[i].a_grad * self.layers[i].activation.diff(self.layers[i].h[:, batch*self.batch_size:(batch+1)*self.batch_size])
                    # print("Layer -", i)
                    # print("a_grad shape:", self.layers[i].a_grad.shape)
                    # print("h_grad shape:", self.layers[i].h_grad.shape)
                    # print("Additional:", self.layers[i-1].a.T.shape)
                    self.layers[i].W_grad = self.layers[i].h_grad @ self.layers[i-1].a[:, batch*self.batch_size:(batch+1)*self.batch_size].T
                    # self.layers[i].b_grad = self.layers[i].h_grad
                    # print("W_grad shape:", self.layers[i].W_grad.shape)
                    # print("h_grad shape:", self.layers[i].h_grad.shape)
                    # print()
                    self.layers[i].W_update = self.layers[i].optimizer.get_update(self.layers[i].W_grad)
                    # self.layers[i].b_update = self.layers[i].b_optimizer.get_update(self.layers[i].b_grad)

                # Update the weights
                for layer in self.layers[1:]:
                    layer.W = layer.W - layer.W_update
                    # layer.b = layer.b - self.b_update
                
                self.forward_propogation()

            if flag == 1:
                break