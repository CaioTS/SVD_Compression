#%%
#Import necessary libraries
from datetime import datetime
import numpy as np
import tensorflow as tf
from datetime import datetime
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, UpSampling2D, Dropout,BatchNormalization,GlobalAveragePooling2D
from keras.utils import to_categorical

from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops

import matplotlib.pyplot as plt

import random
# import math
from sklearn.metrics import accuracy_score, precision_score, recall_score

from functions import *

#%%
tf.keras.utils.get_custom_objects().clear()

@tf.keras.utils.register_keras_serializable(package='Custom', name='L2L0_Reg')
class L2L0_Reg(tf.keras.regularizers.Regularizer):

  def __init__(self, l0=0., beta=0, l2=0.):  # pylint: disable=redefined-outer-name
    self.l0 = K.cast_to_floatx(l0)
    self.beta = K.cast_to_floatx(beta)
    self.l2 = K.cast_to_floatx(l2)

  def __call__(self, x):
    # ones_tensor = tf.ones(x.shape)
    if not self.l2 and not self.l0:
      return K.constant(0.)
    regularization = 0.
    if self.l0:
      ones_tensor = tf.ones(x.shape)
      regularization += self.l0 * math_ops.reduce_sum(ones_tensor-math_ops.exp(-self.beta*math_ops.abs(x)))
      # regularization += self.l0 * math_ops.reduce_sum( tf.clip_by_value(self.beta * math_ops.abs(x), 0, 1)  )
    if self.l2:
      regularization += self.l2 * math_ops.reduce_sum(math_ops.square(x))
    return regularization

  def get_config(self):
    return {'l0': float(self.l0), 'beta': float(self.beta), 'l2': float(self.l2)}

  @classmethod
  def from_config(cls, config):
      l0 = float(config.pop("l0"))
      beta = float(config.pop("beta"))
      l2 = float(config.pop("l2"))
      return cls(l0=l0,beta=beta,l2=l2)


def l0reg(l0=0.0001, l2=0.001, beta=10):
  return L2L0_Reg(l0=l0, beta=beta, l2=l2)

#%%
# Load Unregularized and LOL2 Models
arqs = ["lenetsemreg.keras","lenetL0L2.keras"]

for arq in arqs:
    model = keras.models.load_model(arq)
    print(model.summary())
    for k in range(1,4):
        plt.hist(model.layers[k].get_weights()[0].flatten(),bins=100)
        plt.title(f"Layer {k} - {arq}")
        plt.show() 
#%%
model     = tf.keras.models.load_model('lenetsemreg.keras')
model_reg = tf.keras.models.load_model('lenetL0L2.keras')

#%%
from functions import * 
# %%
X_train, y_train, X_test, y_test = load_mnist()
print('No Regularization Accuracy: ', model.evaluate(X_test, y_test, verbose=0)[1])
print('L0L2 Regularization Accuracy: ', model_reg.evaluate(X_test, y_test, verbose=0))
# %%
#S Vector for both models
U, S, VT = [],[],[] 

weights = []
biases  = []

weights_L2L0 = []
biases_L2L0  = []

for layer in model.layers:
    # Check if the layer has weights
    if len(layer.get_weights()) > 0:
        weights.append(layer.get_weights()[0])  # Extract the weight matrix
        biases.append(layer.get_weights()[1])

for layer in model_reg.layers:
    # Check if the layer has weights
    if len(layer.get_weights()) > 0:
        weights_L2L0.append(layer.get_weights()[0])  # Extract the weight matrix
        biases_L2L0.append(layer.get_weights()[1])

for weight in (weights + weights_L2L0):
    t_U , t_S , t_VT = np.linalg.svd(weight)
    U.append(t_U)
    S.append(t_S)
    VT.append(t_VT)
print(f"{len(S[0]) =}, {len(S[1]) = }, {len(S[2]) = }, {len(S[3]) = }, {len(S[4]) = }, {len(S[5]) = }")
# Normalize the x-axis to percentages for each array
x_1 = np.linspace(0, 100, len(S[0]))  # Normalized x-axis for array_1
x_2 = np.linspace(0, 100, len(S[1]))  # Normalized x-axis for array_2
x_3 = np.linspace(0, 100, len(S[2]))  # Normalized x-axis for array_3

# Create a figure with subplots (3 subplots for each color)
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Plot for red color (Rank Layer 1 and Rank Layer 1 - L2L0)
axs[0].plot(x_1, S[0]/max(S[0]), label='Rank Layer 1', color='red')
axs[0].plot(x_1, S[3]/max(S[3]), label='Rank Layer 1 - L2L0', color='blue')
axs[0].set_ylabel('Value')
axs[0].legend(loc='upper right')
axs[0].set_title('Rank Layer 1 ')

# Plot for green color (Rank Layer 2 and Rank Layer 2 - L2L0)
axs[1].plot(x_2, S[1]/max(S[1]), label='Rank Layer 2', color='red')
axs[1].plot(x_2, S[4]/max(S[4]), label='Rank Layer 2 - L2L0', color='blue')
axs[1].set_ylabel('Value')
axs[1].legend(loc='upper right')
axs[1].set_title('Rank Layer 2 ')

# Plot for blue color (Rank Layer 3 and Rank Layer 3 - L2L0)
axs[2].plot(x_3, S[2]/max(S[2]), label='Rank Layer 3', color='red')
axs[2].plot(x_3, S[5]/max(S[5]), label='Rank Layer 3 - L2L0', color='blue')
axs[2].set_xlabel('X-Axis')
axs[2].set_ylabel('Value')
axs[2].legend(loc='upper right')
axs[2].set_title('Rank Layer 3')

# Adjust layout for better spacing
plt.tight_layout()
plt.show()

#%%
#Test for Each Layer
accuracies = []

#Model Unregularized
#Layer 1
accuracies_l1 = []
loss_l1 = []
for i in range(weights[0].shape[1]):
    SVD_set_model_weights(model,weights,biases,(i,0,0))
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    accuracies_l1.append(acc)
    loss_l1.append(loss)
accuracies.append(accuracies_l1)


print(loss_l1[0],accuracies_l1[0])
#%%
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

#MODEL L2L0
accuracies_l1_L2LO = []
for i in range(weights[0].shape[1]):
    SVD_set_model_weights(model_reg,weights_L2L0,biases_L2L0,(i,0,0))
    _, acc = model_reg.evaluate(X_test, y_test, verbose=0)
    accuracies_l1_L2LO.append(acc)
accuracies.append(accuracies_l1_L2LO)

#Layer 2
accuracies_l2_L2LO = []
for i in range(weights[1].shape[1]):
    SVD_set_model_weights(model_reg,weights_L2L0,biases_L2L0,(0,i,0))
    _, acc = model_reg.evaluate(X_test, y_test, verbose=0)
    accuracies_l2_L2LO.append(acc)
accuracies.append(accuracies_l2_L2LO)
#Layer 3
accuracies_l3_L2LO = []
for i in range(weights[2].shape[1]):
    SVD_set_model_weights(model_reg,weights_L2L0,biases_L2L0,(0,0,i))
    _, acc = model_reg.evaluate(X_test, y_test, verbose=0)
    accuracies_l3_L2LO.append(acc)
accuracies.append(accuracies_l3_L2LO)
#Get best Result for each layer and check for a combination of them

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

# Plotting on each subplot
axes[0].plot(accuracies[0],label='Accuracy Unregularized', color='r')
axes[0].plot(accuracies[3],label='Accuracy L2L0', color='b')
axes[0].plot(accuracies[0][0] - np.array(accuracies[0]),label= "Error Unregularized", color='y')
axes[0].plot(accuracies[3][0] - np.array(accuracies[3]),label= "Error L2L0", color='g')
axes[0].set_ylabel   ("Accuracy")
axes[0].set_xlabel("First Layer Ranks Decomposed")
axes[0].legend()

axes[1].plot(accuracies[1] ,label='Accuracy Unregularized', color='r')
axes[1].plot(accuracies[4],label='Accuracy L2L0', color='b')
axes[1].plot(accuracies[1][0] - np.array(accuracies[1]),label= "Error Unregularized", color='y')
axes[1].plot(accuracies[4][0] - np.array(accuracies[4]),label= "Error L2L0", color='g')
axes[1].set_ylabel("Accuracy")
axes[1].set_xlabel("Second Layer Ranks Decomposed")
axes[1].legend()

axes[2].plot(accuracies[2] ,label='Accuracy Unregularized', color='r')
axes[2].plot(accuracies[5],label='Accuracy L2L0', color='b')
axes[2].plot(accuracies[2][0]- np.array(accuracies[2]),label= "Error Unregularized", color='y')
axes[2].plot(accuracies[5][0]- np.array(accuracies[5]),label= "Error L2L0", color='g')
axes[2].set_ylabel("Accuracy")
axes[2].set_xlabel("Third Layer Ranks Decomposed")
axes[2].legend()
#%%
print("U - sizes:",U[0].shape,U[1].shape,U[2].shape,U[3].shape,U[4].shape,U[5].shape)
print("V - sizes:",VT[0].shape,VT[1].shape,VT[2].shape,VT[3].shape,VT[4].shape,VT[5].shape)

# Get 10 first colluns and plot them 

x_U1 = np.linspace(0, 100, len(U[0])) 
x_U2 = np.linspace(0, 100, len(U[1])) 
x_U3 = np.linspace(0, 100, len(U[2])) 

x_V1 = x_U2
x_V2 = x_U3
x_V3 = np.linspace(0,100,len(VT[3]))  

# Create the subplot grid
fig, axs = plt.subplots(10, 2, figsize=(10, 15))

# Plot the first 10 columns of U in the first column of subplots
for i in range(10):
    axs[i, 0].plot(U[0][:,i], label=f'U Column {i+1}', color='blue')
    axs[i, 0].set_ylabel(f'Column {i+1}')
    axs[i, 0].legend()

# Plot the first 10 columns of VT in the second column of subplots
for i in range(10):
    axs[i, 1].plot(VT[0][:,i], label=f'VT Column {i+1}', color='red')
    axs[i, 1].set_ylabel(f'Column {i+1}')
    axs[i, 1].legend()

# Set the xlabel for the last row to avoid overlap
for ax in axs[-1, :]:
    ax.set_xlabel('Index')
fig.suptitle('First 10 Columns of U and VT First Layer - Unregularized', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.95)  # Adjust the top to make space for the suptitle
plt.show()

# %%
# Create the subplot grid
fig, axs = plt.subplots(10, 2, figsize=(10, 15))

# Plot the first 10 columns of U in the first column of subplots
for i in range(10):
    axs[i, 0].plot(U[3][:,i], label=f'U Column {i+1}', color='blue')
    axs[i, 0].set_ylabel(f'Column {i+1}')
    axs[i, 0].legend()

# Plot the first 10 columns of VT in the second column of subplots
for i in range(10):
    axs[i, 1].plot(VT[3][:,i], label=f'VT Column {i+1}', color='red')
    axs[i, 1].set_ylabel(f'Column {i+1}')
    axs[i, 1].legend()

# Set the xlabel for the last row to avoid overlap
for ax in axs[-1, :]:
    ax.set_xlabel('Index')

fig.suptitle('First 10 Columns of U and VT First Layer - L2L0', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.95)  # Adjust the top to make space for the suptitle
plt.show()


# %%
import pandas as pd
import importlib
import functions

importlib.reload(functions)

df_l1 = pd.DataFrame()
df_l2 = pd.DataFrame()
df_l3 = pd.DataFrame()


df_l1["Ranks"] = range(1,len(accuracies[0])+1)
df_l1["No Reg Acc"] = accuracies[0]
df_l1["L2L0 Acc"] = accuracies[3]
df_l1["Compression"] = functions.calc_compression_per_layer(784,300,1)
df_l1["No Reg Acc Error"] = accuracies[0][0] - np.array(accuracies[0][:])

df_l2["Ranks"] = range(1,len(accuracies[1])+1)
df_l2["No Reg Acc"] = accuracies[1]
df_l2["L2L0 Acc"] = accuracies[4]
df_l2["Compression"] = functions.calc_compression_per_layer(300,100,1)
df_l2["No Reg Acc Error"] = accuracies[1][0] - np.array(accuracies[1][:])

df_l3["Ranks"] = range(1,len(accuracies[2])+1)
df_l3["No Reg Acc"] = accuracies[2]
df_l3["L2L0 Acc"] = accuracies[5]
df_l3["Compression"] = functions.calc_compression_per_layer(100,10,1)
df_l3["No Reg Acc Error"] = accuracies[2][0] - np.array(accuracies[2][:])

#%%
plt.plot(df_l1['No Reg Acc'])
# Filter the DataFrame
filtered_df = df_l1.loc[df_l1['No Reg Acc Error'] < 0.01]

# Get the last valid point
last_x = filtered_df.index[-1]  # X-axis (index or another column)
last_y = filtered_df['No Reg Acc'].iloc[-1]  # Y-axis (last 'No Reg Acc' value)

# Plot the last point
plt.scatter(last_x, last_y, color='red', label=f"Delta Acc = {filtered_df['No Reg Acc Error'].iloc[-1]:.2f} @ Compression = {filtered_df['Compression'].iloc[-1]:.2f}")
plt.title("Accuracy x Rank Layer 1")
plt.legend()
plt.show()
#%%
plt.plot(df_l2['No Reg Acc'])
# Filter the DataFrame
filtered_df = df_l2.loc[df_l2['No Reg Acc Error'] < 0.01]

# Get the last valid point
last_x = filtered_df.index[-1]  # X-axis (index or another column)
last_y = filtered_df['No Reg Acc'].iloc[-1]  # Y-axis (last 'No Reg Acc' value)

# Plot the last point
plt.scatter(last_x, last_y, color='red', label=f"Delta Acc = {filtered_df['No Reg Acc Error'].iloc[-1]:.2f} @ Compression = {filtered_df['Compression'].iloc[-1]:.2f}")
plt.title("Accuracy x Rank Layer 2")
plt.legend()
plt.show()
# %%
plt.plot(df_l3['No Reg Acc'])
# Filter the DataFrame
filtered_df = df_l3.loc[df_l3['No Reg Acc Error'] < 0.01]

# Get the last valid point
last_x = filtered_df.index[-1]  # X-axis (index or another column)
last_y = filtered_df['No Reg Acc'].iloc[-1]  # Y-axis (last 'No Reg Acc' value)

# Plot the last point
plt.scatter(last_x, last_y, color='red', label=f"Delta Acc = {filtered_df['No Reg Acc Error'].iloc[-1]:.2f} @ Compression = {filtered_df['Compression'].iloc[-1]:.2f}")
plt.title("Accuracy x Rank Layer 3")

plt.legend()
plt.show()

# %%
#Analysys of LENETL0L2_lnv

arqs = ["lenetsemreg.keras","lenetL0L2Inv.keras"]

for arq in arqs:
    model = keras.models.load_model(arq)
    print(model.summary())
    for k in range(1,4):
        plt.hist(model.layers[k].get_weights()[0].flatten(),bins=100)
        plt.title(f"Layer {k} - {arq}")
        plt.show() 
#%%
model     = tf.keras.models.load_model('lenetsemreg.keras')
model_reg = tf.keras.models.load_model('lenetL0L2Inv.keras')

#%%
from functions import * 
# %%
X_train, y_train, X_test, y_test = load_mnist()
print('No Regularization Accuracy: ', model.evaluate(X_test, y_test, verbose=0)[1])
print('L0L2 Regularization Accuracy: ', model_reg.evaluate(X_test, y_test, verbose=0)[1])
# %%
#S Vector for both models
U, S, VT = [],[],[] 

weights = []
biases  = []

weights_L2L0 = []
biases_L2L0  = []

for layer in model.layers:
    # Check if the layer has weights
    if len(layer.get_weights()) > 0:
        weights.append(layer.get_weights()[0])  # Extract the weight matrix
        biases.append(layer.get_weights()[1])

for layer in model_reg.layers:
    # Check if the layer has weights
    if len(layer.get_weights()) > 0:
        weights_L2L0.append(layer.get_weights()[0])  # Extract the weight matrix
        biases_L2L0.append(layer.get_weights()[1])

for weight in (weights + weights_L2L0):
    t_U , t_S , t_VT = np.linalg.svd(weight)
    U.append(t_U)
    S.append(t_S)
    VT.append(t_VT)
print(f"{len(S[0]) =}, {len(S[1]) = }, {len(S[2]) = }, {len(S[3]) = }, {len(S[4]) = }, {len(S[5]) = }")
# Normalize the x-axis to percentages for each array
x_1 = np.linspace(0, 100, len(S[0]))  # Normalized x-axis for array_1
x_2 = np.linspace(0, 100, len(S[1]))  # Normalized x-axis for array_2
x_3 = np.linspace(0, 100, len(S[2]))  # Normalized x-axis for array_3

# Create a figure with subplots (3 subplots for each color)
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Plot for red color (Rank Layer 1 and Rank Layer 1 - L2L0Inv)
axs[0].plot(x_1, S[0]/max(S[0]), label='Rank Layer 1', color='red')
axs[0].plot(x_1, S[3]/max(S[3]), label='Rank Layer 1 - L2L0Inv', color='blue')
axs[0].set_ylabel('Value')
axs[0].legend(loc='upper right')
axs[0].set_title('Rank Layer 1 ')

# Plot for green color (Rank Layer 2 and Rank Layer 2 - L2L0Inv)
axs[1].plot(x_2, S[1]/max(S[1]), label='Rank Layer 2', color='red')
axs[1].plot(x_2, S[4]/max(S[4]), label='Rank Layer 2 - L2L0Inv', color='blue')
axs[1].set_ylabel('Value')
axs[1].legend(loc='upper right')
axs[1].set_title('Rank Layer 2 ')

# Plot for blue color (Rank Layer 3 and Rank Layer 3 - L2L0Inv)
axs[2].plot(x_3, S[2]/max(S[2]), label='Rank Layer 3', color='red')
axs[2].plot(x_3, S[5]/max(S[5]), label='Rank Layer 3 - L2L0Inv', color='blue')
axs[2].set_xlabel('X-Axis')
axs[2].set_ylabel('Value')
axs[2].legend(loc='upper right')
axs[2].set_title('Rank Layer 3')

# Adjust layout for better spacing
plt.tight_layout()
plt.show()

#%%
#Test for Each Layer
accuracies = []

#Model Unregularized
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

#MODEL L2L0Inv
accuracies_l1_L2LO = []
for i in range(weights[0].shape[1]):
    SVD_set_model_weights(model_reg,weights_L2L0,biases_L2L0,(i,0,0))
    _, acc = model_reg.evaluate(X_test, y_test, verbose=0)
    accuracies_l1_L2LO.append(acc)
accuracies.append(accuracies_l1_L2LO)

#Layer 2
accuracies_l2_L2LO = []
for i in range(weights[1].shape[1]):
    SVD_set_model_weights(model_reg,weights_L2L0,biases_L2L0,(0,i,0))
    _, acc = model_reg.evaluate(X_test, y_test, verbose=0)
    accuracies_l2_L2LO.append(acc)
accuracies.append(accuracies_l2_L2LO)
#Layer 3
accuracies_l3_L2LO = []
for i in range(weights[2].shape[1]):
    SVD_set_model_weights(model_reg,weights_L2L0,biases_L2L0,(0,0,i))
    _, acc = model_reg.evaluate(X_test, y_test, verbose=0)
    accuracies_l3_L2LO.append(acc)
accuracies.append(accuracies_l3_L2LO)
#Get best Result for each layer and check for a combination of them

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

# Plotting on each subplot
axes[0].plot(accuracies[0],label='Accuracy Unregularized', color='r')
axes[0].plot(accuracies[3],label='Accuracy L2L0Inv', color='b')
axes[0].set_ylabel   ("Accuracy")
axes[0].set_xlabel("First Layer Ranks Decomposed")
axes[0].legend()

axes[1].plot(accuracies[1] ,label='Accuracy Unregularized', color='r')
axes[1].plot(accuracies[4],label='Accuracy L2L0Inv', color='b')
axes[1].set_ylabel("Accuracy")
axes[1].set_xlabel("Second Layer Ranks Decomposed")
axes[1].legend()

axes[2].plot(accuracies[2] ,label='Accuracy Unregularized', color='r')
axes[2].plot(accuracies[5],label='Accuracy L2L0Inv', color='b')
axes[2].set_ylabel("Accuracy")
axes[2].set_xlabel("Third Layer Ranks Decomposed")
axes[2].legend()
# %%
