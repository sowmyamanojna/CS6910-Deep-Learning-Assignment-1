print("Importing packages ... ", end="")
import numpy as np
from activations import Sigmoid, Tanh, Relu, Softmax
from layers import Input, Dense
from optimizers import Normal, Momentum, Nesterov, AdaGrad, RMSProp, Adam, Nadam
from layers import Input, Dense
from network import NeuralNetwork
from loss import CrossEntropy
from helper import OneHotEncoder, MinMaxScaler

import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
print("Done!")

map_optimizers = {"Normal":Normal(), "Momentum":Momentum(), "Nesterov":Nesterov(), "AdaGrad":AdaGrad(), "RMSProp":RMSProp(), "Adam":Adam(), "Nadam":Nadam()}
#################################################################
print("Loading data ... ", end="")
[(x_train, y_train), (x_test, y_test)] = fashion_mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
print("Done!")

print("Size of Training data:", x_train.shape)
print("Size of Validation data:", x_val.shape)

print("Performing Scaling and Encoding transformations on the data ... ", end="")
scaler = MinMaxScaler()
scaler.fit(x_train)
X_scaled = scaler.transform(x_train)
X_val_scaled = scaler.transform(x_train)

X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1]*X_scaled.shape[2]).T
X_val_scaled = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1]*X_val_scaled.shape[2]).T

encoder = OneHotEncoder()
t_full = encoder.fit_transform(y_train, 10)
print("Done!")

X_scaled = X_scaled[:, :100]
t = t_full[:,:100]
#################################################################
layers = [Input(data=X_scaled), 
          Dense(size=64, activation="Sigmoid", name="HL1"), 
          Dense(size=10, activation="Sigmoid", name="OL")]

# model = NeuralNetwork(layers=layers, batch_size=60000, optimizer="Normal", intialization="RandomNormal", epochs=int(5e3), t=t)
model = NeuralNetwork(layers=layers, batch_size=60000, optimizer="Adam", intialization="RandomNormal", loss="CrossEntropy", epochs=int(1000), t=t)
model.forward_propogation()
first_pass_y = model.layers[-1].y
model.backward_propogation()

print("Number Correctly classified in untrained network:", np.sum(np.argmax(first_pass_y, axis=0) == y_train[:100]))
print("Number Correctly classified in trained network:", np.sum(np.argmax((model.layers[-1].y), axis=0) == y_train[:100]))

#################################################################

plt.figure()
plt.plot(np.array(model.accuracy_hist)/100, label="accuracy")
plt.plot(np.array(model.loss_hist)/100, label="loss")
plt.title("Accuracy and Loss of the model")
plt.legend()
plt.grid()
plt.show()