import math
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from copy import deepcopy
from activations import Sigmoid, Tanh, Relu, Softmax
from layers import Input, Dense
from optimizers import Normal, Momentum, Nesterov, AdaGrad, RMSProp, Adam, Nadam
from layers import Input, Dense
from loss import CrossEntropy, SquaredError
from helper import OneHotEncoder

map_optimizers = {"Normal":Normal(), "Momentum":Momentum(), "Nesterov":Nesterov(), "AdaGrad":AdaGrad(), "RMSProp":RMSProp(), "Adam":Adam(), "Nadam":Nadam()}
map_losses = {"SquaredError":SquaredError(), "CrossEntropy":CrossEntropy()}
################################################
#         Network
################################################
class NeuralNetwork():
    def __init__(self, layers, batch_size, optimizer, intialization, epochs, t, loss, X_val=None, t_val=None, optim_params=None):
        self.layers = layers
        self.batch_size = batch_size
        self.intialization = intialization
        self.epochs = epochs
        self.optimizer = optimizer
        self.t = t
        self.num_batches = math.ceil(self.t.shape[1]/batch_size)
        self.loss_type = loss
        self.loss = map_losses[loss]
        if t_val is not None:
            self.X_val = X_val
            self.layers[0].a_val = X_val
            self.t_val = t_val
        self.param_init(optimizer, optim_params)

    def param_init(self, optimizer, optim_params):
        size_prev = self.layers[0].size
        for layer in self.layers[1:]:
            # layer.W_size = (layer.size, size_prev+1)
            layer.W_size = (layer.size, size_prev)
            size_prev = layer.size
            layer.W_optimizer = deepcopy(map_optimizers[optimizer])
            layer.b_optimizer = deepcopy(map_optimizers[optimizer])
            # Code to set params
            if optim_params:
                layer.W_optimizer.set_params(optim_params)
                layer.b_optimizer.set_params(optim_params)

        if self.intialization == "RandomNormal":
            for layer in self.layers[1:]:
                layer.W = np.random.normal(loc=0, scale=1.0, size = layer.W_size)
                layer.b = np.zeros((layer.W_size[0], 1))

        elif self.intialization == "XavierUniform":
            for layer in self.layers[1:]:
                initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)#, seed=42)
                layer.W = np.array(initializer(shape=layer.W_size))
                layer.b = np.zeros((layer.W_size[0], 1))

        elif self.intialization == "Test":
            for layer in self.layers[1:]:
                layer.W = np.ones(layer.W_size)*0.5
                layer.b = np.zeros((layer.W_size[0], 1))


    def forward_propogation(self):
        for i in range(1, len(self.layers)):
            # print("Layer:", i, self.layers[i].W.shape)
            # Pre-activation
            self.layers[i].h = self.layers[i].W @ self.layers[i-1].a - self.layers[i].b
            # Activation
            self.layers[i].a = self.layers[i].activation.value(self.layers[i].h)
            # Validation
            self.layers[i].h_val = self.layers[i].W @ self.layers[i-1].a_val - self.layers[i].b
            self.layers[i].a_val = self.layers[i].activation.value(self.layers[i].h_val)
        
        if self.loss_type == "CrossEntropy":
            # Final sofmax activation
            self.layers[-1].y = Softmax().value(self.layers[-1].a)
            self.layers[-1].y_val = Softmax().value(self.layers[-1].a_val)
        else:
            self.layers[-1].y = self.layers[-1].a
            self.layers[-1].y_val = self.layers[-1].a_val

    def check_test(self, X_test, t_test):
        self.layers[0].a_test = X_test
        for i in range(1, len(self.layers)):
            self.layers[i].h_test = self.layers[i].W @ self.layers[i-1].a_test - self.layers[i].b
            self.layers[i].a_test = self.layers[i].activation.value(self.layers[i].h_test)

        if self.loss=="CrossEntropy":
            self.layers[-1].y_test = Softmax().value(self.layers[-1].a_test)
        else:
            self.layers[-1].y_test = self.layers[-1].a_test

        loss_test = self.loss.calc_loss(t_test, self.layers[-1].y_test)

        encoder = OneHotEncoder()
        y_tmp = encoder.inverse_transform(self.layers[-1].y_test)
        t_tmp = encoder.inverse_transform(t_test)
        acc_test = np.sum(y_tmp==t_tmp)
        return acc_test, loss_test, self.layers[-1].y_test


    def backward_propogation(self):
        # Initialize variables neesed to keep track of loss
        self.eta_hist = []
        self.loss_hist = []
        self.accuracy_hist = []
        self.loss_hist_val = []
        self.accuracy_hist_val = []
        self.loss = SquaredError()
        flag = 0

        # Perform Backprop
        # for _ in range(self.epochs):
        for _ in tqdm(range(self.epochs)):
            for batch in range(self.num_batches):
                # print("\n", "="*50)
                # print("Batch:", batch)
                # X_batch = self.layers[0].input[batch*self.batch_size:(batch+1)*self.batch_size]
                t_batch = self.t[:, batch*self.batch_size:(batch+1)*self.batch_size]
                y_batch = self.layers[-1].y[:, batch*self.batch_size:(batch+1)*self.batch_size]
                self.y_batch = y_batch
                self.t_batch = t_batch

                # Calculate Loss, grad wrt y and softmax for last layer
                # print("t:\n", self.t)
                # print("y:\n", self.layers[-1].y)
                self.eta_hist.append(self.layers[-1].W_optimizer.eta)
                self.loss_hist.append(self.loss.calc_loss(self.t, self.layers[-1].y))
                train_acc, val_acc = self.get_accuracy(validation=True)
                self.accuracy_hist.append(train_acc)
                self.loss_hist_val.append(self.loss.calc_loss(self.t_val, self.layers[-1].y_val))
                self.accuracy_hist_val.append(val_acc)
                # print(self.loss_hist[-1])
                
                try:
                    if self.loss_hist[-1] > self.loss_hist[-2]:
                        for layer in self.layers[1:]:
                            layer.W_optimizer.set_params({"eta":self.optimizer.eta/2})
                            layer.b_optimizer.set_params({"eta":self.optimizer.eta/2})
                        flag = 1
                except:
                    pass

                if flag == 1:
                    break

                # self.layers[-1].cross_grad = self.loss.diff()
                self.layers[-1].a_grad = self.loss.diff(self.t_batch, self.y_batch)
                self.layers[-1].h_grad = self.layers[-1].a_grad * self.layers[-1].activation.diff(self.layers[-1].h[:, batch*self.batch_size:(batch+1)*self.batch_size])

                self.layers[-1].W_grad = self.layers[-1].h_grad @ self.layers[-2].a[:, batch*self.batch_size:(batch+1)*self.batch_size].T
                self.layers[-1].W_update = self.layers[-1].W_optimizer.get_update(self.layers[-1].W_grad)
                
                self.layers[-1].b_grad = -np.sum(self.layers[-1].h_grad, axis=1).reshape(-1,1)
                self.layers[-1].b_update = self.layers[-1].b_optimizer.get_update(self.layers[-1].b_grad)

                # print("Last Layer")
                # print("a_grad shape:", self.layers[-1].a_grad.shape)
                # print("h_grad shape:", self.layers[-1].h_grad.shape)
                # print("W_grad shape:", self.layers[-1].W_grad.shape)
                # print("W_update shape:", self.layers[-1].W_update.shape)
                # print("W_shape:", self.layers[-1].W.shape)
                # print("a_grad:\n", self.layers[-1].a_grad)
                # print("h_grad:\n", self.layers[-1].h_grad)
                # print("W_grad:\n", self.layers[-1].W_grad)

                assert self.layers[-1].W_update.shape == self.layers[-1].W.shape, "Sizes don't match"


                # Backpropogation for the remaining layers
                for i in range(len(self.layers[:-2]), 0, -1):
                    self.layers[i].a_grad = self.layers[i+1].W.T @ self.layers[i+1].h_grad
                    self.layers[i].h_grad = self.layers[i].a_grad * self.layers[i].activation.diff(self.layers[i].h[:, batch*self.batch_size:(batch+1)*self.batch_size])
                    # print("Layer -", i)
                    # print("a_grad shape:", self.layers[i].a_grad.shape)
                    # print("h_grad shape:", self.layers[i].h_grad.shape)

                    # print("Layer -", i)
                    # print("a_grad:", self.layers[i].a_grad)
                    # print("h_grad:", self.layers[i].h_grad)

                    self.layers[i].b_grad = -np.sum(self.layers[i].h_grad, axis=1).reshape(-1,1)
                    self.layers[i].W_grad = self.layers[i].h_grad @ self.layers[i-1].a[:, batch*self.batch_size:(batch+1)*self.batch_size].T
                    
                    # print("W_grad shape:", self.layers[i].W_grad.shape)
                    # print("W_grad:", self.layers[i].W_grad)
                    # print()
                    self.layers[i].W_update = self.layers[i].W_optimizer.get_update(self.layers[i].W_grad)
                    self.layers[i].b_update = self.layers[i].b_optimizer.get_update(self.layers[i].b_grad)
                    # self.layers[i].b_update = self.layers[i].b_optimizer.get_update(self.layers[i].b_grad)

                # Update the weights
                for _, layer in enumerate(self.layers[1:]):
                    layer.W = layer.W - layer.W_update
                    layer.b = layer.b - layer.b_update
                    # print("Layer -", idx)
                    # print("W:\n", layer.W)
                    # print("h:\n", layer.h)

                    # layer.b = layer.b - self.b_update
                # print("Y:\n", self.layers[-1].y)
                self.forward_propogation()

            if flag == 1:
                break

    def describe(self):
        print("Model with the following layers:")
        for i in self.layers:
            print(i)
        print("Loss:", self.loss)
        print("Epochs:", self.epochs)
        print("Batch Size:", self.batch_size)
        print("Optimizer:", self.optimizer)
        print("Initialization:", self.intialization)

    def get_accuracy(self, validation=False, print_vals=False):
        encoder = OneHotEncoder()
        t_train = encoder.inverse_transform(self.t)
        y_train = encoder.inverse_transform(self.layers[-1].y)
        acc_train = np.sum(t_train==y_train)
        if print_vals:
            print("Train Accuracy:", acc_train)

        if validation:
            t_val = encoder.inverse_transform(self.t_val)
            y_val = encoder.inverse_transform(self.layers[-1].y_val)
            acc_val = np.sum(t_val==y_val)
            if print_vals:
                print("Validation Accuracy:", acc_val)
            return acc_train, acc_val
        return acc_train
