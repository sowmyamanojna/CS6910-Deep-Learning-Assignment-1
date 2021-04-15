# CS6910 Assignment 1
Assignment 1 submission for the course CS6910 Fundamentals of Deep Learning.

Team members: N Sowmya Manojna (BE17B007), Shubham Kashyapi (MM16B027)

---

## Question 1
The code for question 1 can be accessed [here](https://github.com/sowmyamanojna/CS6910-Deep-Learning-Assignment-1/blob/main/Question-1.py). The program, reads the data from `keras.datasets`, picks one example from each class and logs the same to `wandb`.

## Questions 2-4
The neural network is implemented by the class `NeuralNetwork`, present in the `network.py` file.  
### Building a `NeuralNetwork`
An instance of `NeuralNetwork` is as follows:
```Python
model = NeuralNetwork(layers=layers, batch_size=2000, optimizer="Normal", \
                      initialization="RandomNormal", loss="CrossEntropy", \
                      epochs=int(100), t=t, X_val=X_val_scaled, t_val=t_val, \
                      use_wandb=False)
```

It can be implemented by passing the following values:

- **layers**  
    An example of layer is as follows:
    
    ```python
    layers = [
                Input(data=X_scaled), 
                Dense(size=64, activation="Sigmoid", name="HL1"), 
                Dense(size=10, activation="Sigmoid", name="OL")
             ]
    ```

    Here, `Input` and `Dense` are layer classes, that can be accessed in the `layers.py` file.
    - An instance of the class `Input` can be created by passing the input to the function call, as shown above.
    - An instance of the class `Dense` can be created by passing the size of the layer, the activation and an optional name to identify the layer.

- **batch_size**  
    The Batch Size is passed as an integer that determines the size of the mini batch to be taken into consideration.

- **optimizer**  
    The optimizer value is passed as a string, that is internally converted into an instance of the specified optimizer class. The optimizer classes are present inside the file `optimizers.py`. An instance of the class can be created by passing the corresponding parameters:
    + Normal: eta   
        (default: eta=0.01)
    + Momentum: eta, gamma   
        (default: eta=1e-3, gamma=0.9)
    + Nesterov: eta, gamma   
        (default: eta=1e-3, gamma=0.9)
    + AdaGrad: eta, eps   
        (default: eta=1e-2, eps=1e-7)
    + RMSProp: beta, eta , eps    
        (default: beta=0.9, eta = 1e-3, eps = 1e-7)
    + Adam: beta1, beta2, eta, eps   
        (default: beta1=0.9, beta2=0.999, eta=1e-2, eps=1e-8)
    + Nadam: beta1, beta2, eta, eps   
        (default: beta1=0.9, beta2=0.999, eta=1e-3, eps=1e-7)

- **intialization**: A string - `"RandomNormal"` or `"XavierUniform"` can be passed to change the initialization of the weights in the model.

- **epochs**: The number of epochs is passed as an integer to the neural network.

- **t**: `t` is the `OneHotEncoded` matrix of the vector `y_train`, of size (10,n), where n is the number of sample.

- **loss**: The loss type is passed as a string, that is internally converted into an instance of the specified loss class. The optimizer classes are present inside the file `loss.py`. 

- **X_val**: The validation dataset, used to validate the model.

- **t_val**: `t_val` is the `OneHotEncoded` matrix of the vector `y_val`, of size (10,n), where n is the number of sample.

- **use_wandb**: A flag that lets the user choose whether they want to use wandb for the run or not.
 
- **optim_params**: Optimization parameters to be passed to the optimizers.

### Training the `NeuralNetwork`
The model can be trained by calling the member function: `forward_propogation`, followed by `backward_propogation`. It is done as follows:

```python
model.forward_propogation()
model.backward_propogation()
```

### Testing the `NeuralNetwork`
The model can be tested by calling the `check_test` member function, with the testing dataset and the expected `y_test`. The `y_test` values are only used for calculating the test accuracy. It is done in the following manner:

```python
acc_test, loss_test, y_test_pred = model.check_test(X_test_scaled, t_test)
```

## Question 7
The confusion matrix is logged using the following code:

```python
wandb.log({"conf_mat" : wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=y_test[:9000],
                        preds=y_test_pred,
                        class_names=["T-shirt/top","Trouser","Pullover",\
                                     "Dress","Coat","Sandal","Shirt","Sneaker",\
                                     "Bag","Ankle boot"])})
```


## Question 10
Three hyperparameter sets were selected and were run on the MNIST dataset. The configurations choosen are as follows:

- Configuration 1: 
    - `optimizer` = Adam, 
    - `init` = XavierUniform, 
    - `activation` = tanh, 
    - `hidden_layer_size` = 64, 
    - `batch_size` = 1024, 
    - `num_hidden_layers` = 1

- Configuration 2: 
    - `optimizer` = Adam, 
    - `init` = XavierUniform, 
    - `activation` = tanh, 
    - `hidden_layer_size` = 32, 
    - `batch_size` = 128, 
    - `num_hidden_layers` = 1

- Configuration 3: 
    - `optimizer` = Adam, 
    - `init` = XavierUniform, 
    - `activation` = relu, 
    - `hidden_layer_size` = 32, 
    - `batch_size` = 1024, 
    - `num_hidden_layers` = 1

---
The codes are organized as follows:

| Question | Location | Function | 
|----------|----------|----------|
| Question 1 | [Question-1](https://github.com/sowmyamanojna/CS6910-Deep-Learning-Assignment-1/blob/main/Question-1.py) | Logging Representative Images | 
| Question 2 | [Question-2](https://github.com/sowmyamanojna/CS6910-Deep-Learning-Assignment-1/blob/f266f73a9a28c20f3dc26c1902c9aa64bf142912/network.py#L67) | Feedforward Architecture |
| Question 3 | [Question-3](https://github.com/sowmyamanojna/CS6910-Deep-Learning-Assignment-1/blob/main/Question-3.py) | Complete Neural Network |
| Question 4 | [Question-4](https://github.com/sowmyamanojna/CS6910-Deep-Learning-Assignment-1/blob/main/Question-4.py) | Hyperparameter sweeps using `wandb` |
| Question 7 | [Question-7](https://github.com/sowmyamanojna/CS6910-Deep-Learning-Assignment-1/blob/main/Question-7.py) | Confusion Matrix logging for the best Run | 
| Question 10 | [Question-10](https://github.com/sowmyamanojna/CS6910-Deep-Learning-Assignment-1/blob/main/Question-10.py) | Hyperparameter configurations for MNIST data (Q10) | 
