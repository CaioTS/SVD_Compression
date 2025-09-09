#%%%
#Download weights of a trained Lenet

#%%
#Import necessary libraries
import tensorflow as tf
import numpy as np
import math
import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, datasets
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.initializers import RandomNormal
# import math
from sklearn.metrics import accuracy_score, precision_score, recall_score

from functions import *
#%%
def lenet_nn():
	"""
	Function to define the architecture of a neural network model
	following 300 100 Dense Fully-Connected architecture for MNIST
	dataset.
    
	Output: Returns designed and compiled neural network model
	"""
    
	model = Sequential()
	model.add(InputLayer(input_shape=(784, )))
	# model.add(Flatten())
	model.add(
		Dense(
			units = 300, activation='relu',
			kernel_initializer=tf.initializers.GlorotUniform()
			)
		)

	# model.add(l.Dropout(0.2))

	model.add(
		Dense(
			units = 100, activation='relu',
			kernel_initializer=tf.initializers.GlorotUniform()
			)
		)
        
	# model.add(l.Dropout(0.1))

	model.add(
		Dense(
			units = 10, activation='softmax'
			)
		)
	# Compile pruned NN-
	model.compile(
		loss=tf.keras.losses.categorical_crossentropy,
		# optimizer='adam',
		optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0012),
		metrics=['accuracy'])
    
	return model

#%%
#Load Model weights
model = lenet_nn()
model.load_weights('LeNet_300_MNIST_Magnitude_Winning_Ticket_Distribution_91.18900266306589.h5')
model.summary()


#%%
#Extract each layer as a matrix
# first_layer(784x300)
# second_layer(300x100)
# third_layer(100x10)
# Loop through each layer in the model
weights = []
biases  = []
for layer in model.layers:
    # Check if the layer has weights
    if len(layer.get_weights()) > 0:
        weights.append(layer.get_weights()[0])  # Extract the weight matrix
        biases.append(layer.get_weights()[1])
        print(f"Layer: {layer.name}")
        print(f"Weights matrix shape: {weights[-1].shape}")
        print(weights[-1])  # Display the weight matrix
        print("="*50)
    else:
        print(f"Layer {layer.name} does not have weights.")

#%%
# Make SVD Conversion for each layer
U, S, VT = [],[],[] 
for weight in weights:
    t_U , t_S , t_VT = np.linalg.svd(weight)
    U.append(t_U)
    S.append(t_S)
    VT.append(t_VT)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

# Plotting on each subplot
axes[0].plot(S[0]/np.max(S[0]) ,label='S[0]', color='r')
axes[0].set_title('Rank First Layer')
axes[0].legend()

axes[1].plot(S[1]/np.max(S[1]) ,label='S[1]', color='g')
axes[1].set_title('Rank Second Layer')
axes[1].legend()

axes[2].plot(S[2]/np.max(S[2]) ,label='S[2]', color='b')
axes[2].set_title('Rank Third Layer')
axes[2].legend()

# Show the plot
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

#%%
#Load the weights from the rank reductions SVDs to
#the model, so that we are able to evaluate its accuracy
S_matrix = []
r_weights = []

for w,s,u,vt in zip(weights,S,U,VT):
    S_matrix.append(np.zeros_like(w, dtype=float))
    np.fill_diagonal(S_matrix[-1], s)
    r_weights.append(u @ S_matrix[-1] @ vt)
    #Check if it reconstructed the weights correctly
    print("Error for each layer reconstructed:",sum(sum(r_weights[-1] - w)))
#Run Error with different ranks in SVD

weights_list = []
for w , b in zip(r_weights,biases):
    weights_list.append(w)
    weights_list.append(b)

model.set_weights(weights_list)

# %%

# %%

# Load MNIST (https://github.com/arjun-majumdar/Lottery_Ticket_Hypothesis-TensorFlow_2/blob/master/LeNet_MNIST.ipynb)
def load_MNIST():
    num_classes = 10

    img_rows, img_cols = 28, 28
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    if tf.keras.backend.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

    # Convert datasets to floating point types-
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Normalize the training and testing datasets-
    X_train /= 255.0
    X_test /= 255.0

    # convert class vectors/target to binary class matrices or one-hot encoded values-
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # Reshape training and testing sets-
    X_train = X_train.reshape(X_train.shape[0], 784)
    X_test = X_test.reshape(X_test.shape[0], 784)

    return X_test, y_test

#%%
# Evaluate model accuracy
#Load MNIST
X_test, y_test = load_MNIST()

#Load Mode
model = lenet_nn()
model.load_weights('LeNet_300_MNIST_Magnitude_Winning_Ticket_Distribution_91.18900266306589.h5')
weights = []
biases  = []
for layer in model.layers:
    # Check if the layer has weights
    if len(layer.get_weights()) > 0:
        weights.append(layer.get_weights()[0])  # Extract the weight matrix
        biases.append(layer.get_weights()[1])

#Calculate Accuracy for original model
_, model_accuracy = model.evaluate(X_test, y_test, verbose=0)

#Test SVD with no rank decomposition
SVD_set_model_weights(model,weights,biases,(0,0,0))
_, model_SVD_r0_accuracy = model.evaluate(X_test, y_test, verbose=0)
error = abs(model_SVD_r0_accuracy - model_accuracy)

if ( error <1e-7):
    print(f"SVD_set_model_weights Sucess: {error}")

#Test for Each Layer
accuracies = []

#Layer 1
accuracies_l1 = []
for i in range(weights[0].shape[1]):
    SVD_set_model_weights(model,weights,biases,(i,0,0))
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    accuracies_l1.append(acc)
accuracies.append(accuracies_l1)

#Layer 2
accuracies_l2 = []
for i in range(weights[1].shape[1]):
    SVD_set_model_weights(model,weights,biases,(0,i,0))
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    accuracies_l2.append(acc)
accuracies.append(accuracies_l2)
#Layer 3
accuracies_l3 = []
for i in range(weights[2].shape[1]):
    SVD_set_model_weights(model,weights,biases,(0,0,i))
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    accuracies_l3.append(acc)
accuracies.append(accuracies_l3)
#Get best Result for each layer and check for a combination of them

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

# Plotting on each subplot
axes[0].plot(accuracies[0],label='Accuracy', color='r')
axes[0].set_ylabel   ("Accuracy")
axes[0].set_xlabel("First Layer Ranks Decomposed")
axes[0].legend()

axes[1].plot(accuracies[1] ,label='Accuracy', color='g')
axes[1].set_ylabel("Accuracy")
axes[1].set_xlabel("Second Layer Ranks Decomposed")
axes[1].legend()

axes[2].plot(accuracies[2] ,label='accuracy', color='b')
axes[2].set_ylabel("Accuracy")
axes[2].set_xlabel("Third Layer Ranks Decomposed")
axes[2].legend()
#%%
xa_1 = np.linspace(0, 100, len(accuracies[0]))  # Normalized x-axis for array_1
xa_2 = np.linspace(0, 100, len(accuracies[1]))  # Normalized x-axis for array_2
xa_3 = np.linspace(0, 100, len(accuracies[2]))  # Normalized x-axis for array_3

# Add labels and title
plt.plot(xa_1, accuracies[0]/np.max(accuracies[0]), label='Rank Layer 1', color='red')
plt.plot(xa_2, accuracies[1]/np.max(accuracies[1]), label='Rank Layer 2', color='green')
plt.plot(xa_3, accuracies[2]/np.max(accuracies[2]), label='Rank Layer 3', color='blue')
plt.xlabel('Ranks Decomposed (Normalized to Percentage)')
plt.title('Accuracy for each layer')

# Add a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Make SVD Conversion for each layer
U, S, VT = [],[],[] 
for weight in weights:
    t_U , t_S , t_VT = np.linalg.svd(weight)
    U.append(t_U)
    S.append(t_S)
    VT.append(t_VT)

#fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns
#
## Plotting on each subplot
#axes[0].plot(S[0]/np.max(S[0]) ,label='S[0]', color='r')
#axes[0].set_title('Rank First Layer')
#axes[0].legend()
#
#axes[1].plot(S[1]/np.max(S[1]) ,label='S[1]', color='g')
#axes[1].set_title('Rank Second Layer')
#axes[1].legend()
#
#axes[2].plot(S[2]/np.max(S[2]) ,label='S[2]', color='b')
#axes[2].set_title('Rank Third Layer')
#axes[2].legend()

# Show the plot
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Normalize the x-axis to percentages for each array
x_1 = np.linspace(0, 100, len(S[0]))  # Normalized x-axis for array_1
x_2 = np.linspace(0, 100, len(S[1]))  # Normalized x-axis for array_2
x_3 = np.linspace(0, 100, len(S[2]))  # Normalized x-axis for array_3

# Plot each array
plt.plot(x_1, S[0]/np.max(S[0]), label='Rank Layer 1', color='red')
plt.plot(x_2, S[1]/np.max(S[1]), label='Rank Layer 2', color='green')
plt.plot(x_3, S[2]/np.max(S[2]), label='Rank Layer 3', color='blue')

# Add labels and title
plt.xlabel('Ranks Decomposed (Normalized to Percentage)')
plt.title('S Array of all layers')

# Add a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
#Calcular número de operações necessárias para Camada Original e com redução de rank para cada camada
# Layer X (m x n)

# Layer 1 (784x300)
print(U[0].shape, S[0].shape, VT[0].shape) 
# Layer 2 (300x100)

# Layer 3 (1000x10)


# %%
