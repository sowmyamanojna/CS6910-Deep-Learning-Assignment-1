# To add a new cell, type "# %%"
# To add a new markdown cell, type "# %% [markdown]"
####################################################################
import wandb
import numpy as np
import os
from activations import Sigmoid, Tanh, Relu, Softmax
from layers import Input, Dense
from optimizers import Normal, Momentum, Nesterov, AdaGrad, RMSProp, Adam, Nadam
from network import NeuralNetwork
from loss import CrossEntropy, SquaredError
from helper import OneHotEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split

from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

#################################################################### [markdown]
# # Loss on Training Data
####################################################################

print("Loading data ... ", end="")
[(x_train, y_train), (x_test, y_test)] = fashion_mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
print("Done!")

print("Size of Training data:", x_train.shape)
print("Size of Validation data:", x_val.shape)

print("Performing Scaling and Encoding transformations on the data ... ", end="")
scaler = MinMaxScaler()
scaler.fit(x_train)
X_scaled = x_train/255
X_val_scaled = x_val/255
X_test_scaled = x_test/255

X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1]*X_scaled.shape[2]).T
X_val_scaled = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1]*X_val_scaled.shape[2]).T
X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1]*X_test_scaled.shape[2]).T

encoder = OneHotEncoder()
t = encoder.fit_transform(y_train, 10)
t_val = encoder.fit_transform(y_val, 10)
t_test = encoder.fit_transform(y_test, 10)
print("Done!")

X_scaled = X_scaled[:, :21000]
X_test_scaled = X_test_scaled[:, :9000]
t = t[:, :21000]
t_test = t_test[:, :9000]
####################################################################
# # Preparing small dataset to test the code
# [(X_train, y_train), (X_test, y_test)] = fashion_mnist.load_data()
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X_train)
# X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1]*X_scaled.shape[2]).T

# encoder = OneHotEncoder()
# t = encoder.fit_transform(y_train, 10)


####################################################################
sweep_config = {"name": "random-test-sweep", "method": "grid"}
sweep_config["metric"] = {"name": "loss", "goal": "minimize"}
parameters_dict = {
                "num_epochs": {"values": [10, 50, 100]}, \
                # "num_hidden_layers": {"values": [3, 4, 5]}, \
                "size_hidden_layer": {"values": [32, 64, 128]}, \
                "learning_rate": {"values": [1e-3, 1e-4]}, \
                "optimizer": {"values": ["Normal","Momentum","AdaGrad","RMSProp","Adam", "Nadam"]}, \
                "batch_size": {"values": [128, 256, 512, 1024, 10000, 60000]}, \
                "weight_init": {"values": ["RandomNormal", "XavierUniform"]} , \
                "activation": {"values": ["Sigmoid", "Tanh", "Relu"]}, \
                "loss": {"values": ["CrossEntropy", "SquaredError"]}, \
                  }
sweep_config["parameters"] = parameters_dict
for i in sweep_config:
    print(i, sweep_config[i])

####################################################################
def train_nn(config = sweep_config):
    with wandb.init(config = config):
        config = wandb.init().config
        wandb.run.name = "epochs_{}_layersize_{}_opt_{}_batch_{}_init_{}".format(config.num_epochs,                                                                             config.size_hidden_layer,                                                                             config.optimizer,                                                                             config.batch_size,                                                                             config.weight_init)

        layers = [Input(data=X_scaled),\
                  Dense(size=config.size_hidden_layer, activation=config.activation, name="HL1"),\
                  Dense(size=10, activation="Sigmoid", name="OL")]

        nn_model = NeuralNetwork(layers=layers, batch_size=config.batch_size, \
                                 optimizer=config.optimizer, intialization=config.weight_init, \
                                 epochs=config.num_epochs, t=t, X_val=X_val_scaled, \
                                 t_val=t_val, loss=config.loss)

        nn_model.forward_propogation()
        nn_model.backward_propogation()
        acc_val, loss_val, _ = nn_model.check_test(X_val_scaled, t_val)
        acc_test, loss_test, _ = nn_model.check_test(X_test_scaled, t_test)
        
        wandb.log({"val_loss_end": loss_val/t_val.shape[1], \
                   "val_acc_end": acc_val/t_val.shape[1], \
                   "test_loss_end": loss_test/t_test.shape[1], \
                   "test_acc_end": acc_test/t_test.shape[1], \
                   "epoch":config.num_epochs})

        for step_loss in nn_model.loss_hist:
            wandb.log({'loss': step_loss/t.shape[1]})

        for step_acc in nn_model.accuracy_hist:
            wandb.log({'accuracy': step_acc/t.shape[1]})
        
        for step_val_loss in nn_model.loss_hist_val:
            wandb.log({'val_loss': step_val_loss/t_val.shape[1]})

        for step_val_accuracy in nn_model.accuracy_hist_val:
            wandb.log({'val_accuracy': step_val_accuracy/t_val.shape[1]})
####################################################################
sweep_id = wandb.sweep(sweep_config, project = "CS6910-Assignment-1")
wandb.agent(sweep_id, function = train_nn)

#################################################################### [markdown]
