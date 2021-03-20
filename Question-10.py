#!/usr/bin/env python
# coding: utf-8

###############################################################
import wandb
import numpy as np
import os
from activations import Sigmoid, Tanh, Relu, Softmax
from layers import Input, Dense
from optimizers import Momentum, Nesterov, AdaGrad, RMSProp, Adam, Nadam
from network import NeuralNetwork
from loss import CrossEntropy
from helper import OneHotEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split

from keras.datasets import mnist
import matplotlib.pyplot as plt


###############################################################
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

X_scaled = X_train/255
X_test_scaled = X_test/255

X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1]*X_scaled.shape[2]).T
X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1]*X_test_scaled.shape[2]).T

encoder = OneHotEncoder()
t_train = encoder.fit_transform(y_train, 10)
t_test = encoder.fit_transform(y_test, 10)

###############################################################
# Configuration 1: optimizer = Adam, init = XavierUniform, activation = tanh, hidden_layer_size = 64, batch_size = 1024,  num_hidden_layers = 1
###############################################################
layers = [
		  Input(data=X_scaled),\
		  Dense(size=64, activation='Tanh', name='HL1'),\
		  Dense(size=10, activation='Sigmoid', name='OL')
	     ]

nn_model1 = NeuralNetwork(layers=layers, batch_size=1024,\
					      optimizer='Adam', initialization='XavierUniform',\
                          epochs=10, t=t_train, X_val=X_test_scaled, \
                          t_val=t_test, loss="CrossEntropy") 
nn_model1.forward_propogation()
nn_model1.backward_propogation()

###############################################################
acc_test, _, _ = nn_model1.check_test(X_test_scaled, y_test)
print('Accuracy on test set = {}'.format(acc_test)

###############################################################
# ## Configuration 2: optimizer = Adam, init = XavierUniform, activation = tanh, hidden_layer_size = 32, batch_size = 128, num_hidden_layers = 1
###############################################################
layers = [
		  Input(data=X_scaled),\
		  Dense(size=32, activation='Tanh', name='HL1'),\
		  Dense(size=10, activation='Sigmoid', name='OL')
	     ]

nn_model2 = NeuralNetwork(layers=layers, batch_size=128,\
					      optimizer='Adam', initialization='XavierUniform',\
                          epochs=10, t=t_train, X_val=X_test_scaled, \
                          t_val=t_test, loss="CrossEntropy") 
nn_model2.forward_propogation()
nn_model2.backward_propogation()

###############################################################
acc_test2, _, _ = nn_model2.check_test(X_test_scaled, y_test)
print('Accuracy on test set = {}'.format(acc_test2)

###############################################################
# Configuration 3: optimizer = Adam, init = XavierUniform, activation = relu, hidden_layer_size = 32, batch_size = 1024, num_hidden_layers = 1
###############################################################
layers = [
		  Input(data=X_scaled),\
		  Dense(size=32, activation='Relu', name='HL1'),\
		  Dense(size=10, activation='Sigmoid', name='OL')
	     ]

nn_model3 = NeuralNetwork(layers=layers, batch_size=1024,\
					      optimizer='Adam', initialization='XavierUniform',\
                          epochs=10, t=t_train, X_val=X_test_scaled, \
                          t_val=t_test, loss="CrossEntropy") 
nn_model3.forward_propogation()
nn_model3.backward_propogation()

###############################################################
acc_test3, _, _ = nn_model3.check_test(X_test_scaled, y_test)
print('Accuracy on test set = {}'.format(acc_test3)

