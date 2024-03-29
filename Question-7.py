print("Importing Libraries ...")
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

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import seaborn as sns
print("Done!")
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
sweep_config = {"name": "best-sweep", "method": "grid"}
sweep_config["metric"] = {"name": "loss", "goal": "minimize"}
parameters_dict = {
                "num_epochs": {"values": [1]}, \
                "size_hidden_layer": {"values": [64]}, \
                "optimizer": {"values": ["RMSProp"]}, \
                "batch_size": {"values": [128]}, \
                "weight_init": {"values": ["XavierUniform"]} , \
                "activation": {"values": ["Sigmoid"]}, \
                "loss": {"values": ["SquaredError"]}, \
                  }
sweep_config["parameters"] = parameters_dict

####################################################################
def train_nn(config = sweep_config):
    with wandb.init(config = config):
        config = wandb.init().config
        wandb.run.name = "e_{}_hl_{}_opt_{}_bs_{}_init_{}_ac_{}_loss_{}".format(config.num_epochs,\
                                                                      config.size_hidden_layer,\
                                                                      config.optimizer,\
                                                                      config.batch_size,\
                                                                      config.weight_init,\
                                                                      config.activation,\
                                                                      config.loss)

        layers = [Input(data=X_scaled),\
                  Dense(size=config.size_hidden_layer, activation=config.activation, name="HL1"),\
                  Dense(size=10, activation=config.activation, name="OL")]

        nn_model = NeuralNetwork(layers=layers, batch_size=config.batch_size, \
                                 optimizer=config.optimizer, initialization=config.weight_init, \
                                 epochs=config.num_epochs, t=t, X_val=X_val_scaled, \
                                 t_val=t_val, loss=config.loss, use_wandb=True)

        nn_model.forward_propogation()
        nn_model.backward_propogation()
        acc_val, loss_val, _ = nn_model.check_test(X_val_scaled, t_val)
        acc_test, loss_test, y_test_pred = nn_model.check_test(X_test_scaled, t_test)
        
        wandb.log({"val_loss_end": loss_val/t_val.shape[1], \
                   "val_acc_end": acc_val/t_val.shape[1], \
                   "test_loss_end": loss_test/t_test.shape[1], \
                   "test_acc_end": acc_test/t_test.shape[1], \
                   "epoch":config.num_epochs})

        # cf_matrix = confusion_matrix(y_test[:9000], y_test_pred)
        # # plt.figure()
        # sns.heatmap(cf_matrix)
        # plt.title("Confusion Matrix")
        # wandb.log({"Confusion Matrix": plt})

        data = [[x, y] for (x, y) in zip(y_test[:9000], y_test_pred)]
        table = wandb.Table(data=data, columns = ["true", "predicted"])

        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=y_test[:9000],
                        preds=y_test_pred,
                        class_names=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"])})

####################################################################
sweep_id = wandb.sweep(sweep_config, project = "trail-1")
wandb.agent(sweep_id, function = train_nn)
#################################################################### [markdown]