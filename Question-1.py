print("Importing packages... ", end="")
##############################################################################
import wandb
import numpy as np
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

wandb.init(project="trail-1")
print("Done!")
##############################################################################
print("Loading data... ", end="")
# Load the dataset
[(x_train, y_train), (x_test, y_test)] = fashion_mnist.load_data()

# Get the number of classes and their name mappings
num_classes = 10
class_mapping = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}
print("Done!")

##############################################################################
# Plotting a figure from each class
plt.figure(figsize=[12, 5])
img_list = []
class_list = []

for i in range(num_classes):
    position = np.argmax(y_train==i)
    image = x_train[position,:,:]
    plt.subplot(2, 5, i+1)
    plt.imshow(image)
    plt.title(class_mapping[i])
    img_list.append(image)
    class_list.append(class_mapping[i])
    
wandb.log({"Question 1": [wandb.Image(img, caption=caption) for img, caption in zip(img_list, class_list)]})
##############################################################################
