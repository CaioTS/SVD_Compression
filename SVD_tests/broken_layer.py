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
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
# import math
from sklearn.metrics import accuracy_score, precision_score, recall_score
from functions import *
#%%
alphal0 = 1e-5
beta = 20
alphal2 = 1e-4
REG = "L2L0"
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

def LeNet_NN_broken(MT):
    """
    Define LeNet 300-100-10 Dense Fully Connected
    Neural Network for MNIST multi-class classification
    """

    model = keras.Sequential()

    model.add(keras.layers.Input(shape=(28,28)))

    model.add(Flatten())

    model.add(
        Dense(units = MT, activation = None,
              kernel_initializer = tf.initializers.GlorotNormal(),
              input_shape = (784,),
              kernel_regularizer=l0reg(l0=alphal0*np.sqrt(235500/266610),l2=alphal2,beta=beta) if REG != "None" else None
             )
    )
    model.add(
          Dense(units = 300, activation = 'relu',
                kernel_initializer = tf.initializers.GlorotNormal(),
                kernel_regularizer=l0reg(l0=alphal0*np.sqrt(30100/266610),l2=alphal2,beta=beta) if REG != "None" else None     
		  )
	)

    model.add(
        Dense(units = 100, activation = 'relu',
              kernel_initializer = tf.initializers.GlorotNormal(),
              kernel_regularizer=l0reg(l0=alphal0*np.sqrt(30100/266610),l2=alphal2,beta=beta) if REG != "None" else None
             )
    )

    model.add(
        Dense(units = 10, activation = 'softmax', kernel_regularizer=l0reg(l0=alphal0*np.sqrt(1010/266610),l2=alphal2,beta=beta) if REG != "None" else None
             )
    )
    model.compile(
	
  	loss=tf.keras.losses.categorical_crossentropy,
		# optimizer='adam',
		optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0012),
		metrics=['accuracy'])
  
    return model

#%%
model_reg = tf.keras.models.load_model('lenetL0L2.keras')

weights = []
biases = []

for layer in model_reg.layers:
    # Check if the layer has weights
    if len(layer.get_weights()) > 0:
        weights.append(layer.get_weights()[0])  # Extract the weight matrix
        biases.append(layer.get_weights()[1])

U , S , VT = np.linalg.svd(weights[0])
US = np.zeros_like(U)
for i in range(len(S)):
    US[:,i] = U[:,i]*S[i]
US = US[:,:len(S)]
print(U.shape,S.shape,VT.shape, US.shape)

# %%
X_train, y_train, X_test, y_test =  load_mnist()
# %%
test_input = X_train[0,:,:,0].reshape(784)

Y = test_input @ weights[0] 
Y_test = np.zeros_like(Y)
print(test_input.shape,weights[0].shape, Y.shape,US.shape,VT.shape)

for i in range(len(S)):
    USX = test_input @ US[:,i]
    temp = np.outer(USX , VT[i,:]).reshape(300)
    Y_test += temp
plt.plot(Y_test - Y)

# %%
model_reg.summary()
model_broken = LeNet_NN_broken(300)
model_broken.summary()
# %%

def set_model_weights(US,VT,orig_weights,orig_biases,model):
  weights_list = []
  zero_bias = np.zeros(US.shape[1])
  zero_bias_2 = np.zeros(VT.shape[1]) 
    #First Layer
  weights_list.append(US)
  weights_list.append(zero_bias)  
  #Second Layer 
  weights_list.append(VT)
  weights_list.append(orig_biases[0])

  #Third layer and so on: 

  for w , b in zip(orig_weights[1:],orig_biases[1:]):
    weights_list.append(w)
    weights_list.append(b)

  model.set_weights(weights_list)

# %%
set_model_weights(US,VT,weights,biases,model_broken)
print(y_test[:10])
# %%
print(X_test.shape)
y_broken = model_broken.predict(X_test)
y_reg = model_reg.predict(X_test)

error = y_broken[:,0] - y_reg[:,0]
print(y_broken.shape)
plt.plot(y_broken[:,0])
#%%
plt.plot(error)
#%%
print(model_broken.evaluate(X_test,y_test))
print(model_reg.evaluate(X_test,y_test))
# %%
Mt = 300
model_broken_rank = LeNet_NN_broken(Mt)
model_broken_rank.summary()
set_model_weights(US[:,:Mt],VT[:,:Mt],weights,biases,model_broken_rank)
print(model_broken_rank.evaluate(X_test,y_test))
print(model_reg.evaluate(X_test,y_test))
# %%
acc = []
loss = []
for i in range(1,301):
  Mt = i
  model_broken_rank = LeNet_NN_broken(Mt)
  set_model_weights(US[:,:Mt],VT[:Mt,:],weights,biases,model_broken_rank)
  [tmp_loss, tmp_acc] = model_broken_rank.evaluate(X_test,y_test)
  acc.append(tmp_acc)
  loss.append(tmp_loss)

[loss_full_model, acc_full_model] = model_reg.evaluate(X_test,y_test)
#%%
#Print Both Losses to check for same results in prediction:
print(f"Original Model Loss :{loss_full_model}")
print(f"Broken Full Model Loss :{loss[-1]}")
plt.plot(loss)
plt.scatter(300,loss_full_model,color = 'red')
#%%
for i in range(len(acc)):
   if acc[i] > acc_full_model - 0.01:
      break

# %%
plt.plot(acc)
plt.scatter(i+1,acc[i],color= 'red')
Mt = i+1 #Adjust index
plt.title(f"Accuracy x M_t ( Acc:{acc[i]} @ Mt = {Mt})")

model_1percent = LeNet_NN_broken(Mt)
set_model_weights(US[:,:Mt],VT[:Mt,:],weights,biases,model_1percent)
print(calc_num_parameters_compression(model_1percent,model_reg))
print(US.shape,VT.shape)


# %%
#Retrain model to try to regain accuray lost ()
# %%

img_rows, img_cols = 28, 28
num_classes = 10
# tf.keras.utils.disable_interactive_logging()
mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test,Y_test) = mnist.load_data()
# tf.keras.utils.enable_interactive_logging()

if tf.keras.backend.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Convert datasets to floating point types-
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize the training and testing datasets-
X_train /= 255.0
X_test /= 255.0

y_train = tf.keras.utils.to_categorical(Y_train, num_classes)
y_test = tf.keras.utils.to_categorical(Y_test, num_classes)

print(X_train.shape)

train_datagen = ImageDataGenerator()
train_datagen.fit(X_train)

valid_datagen = ImageDataGenerator()
valid_datagen.fit(X_train)

print(X_train.shape)

#%%
EPOCHS = 50
rseed = random.randint(1,1000)
try:
  result = model_1percent.fit(
      train_datagen.flow(X_train, y_train, batch_size = 64,seed=rseed,shuffle=True),
      validation_data=valid_datagen.flow(X_test, y_test, batch_size = 64,seed=rseed,shuffle=True),
      epochs = EPOCHS,
      verbose = 2,
      validation_freq = 5
  )
except KeyboardInterrupt:
  print("\n\nParalizado!\n\n")

#%%
print(model_1percent.evaluate(X_test,y_test))
print(model_reg.evaluate(X_test,y_test))
# %%
#Train 5percent error model
for i in range(len(acc)):
   if acc[i] > acc_full_model - 0.05:
      break
print(i+1)
model_5percent = LeNet_NN_broken(i+1)
set_model_weights(US[:,:i+1],VT[:i+1,:],weights,biases,model_5percent)
print(calc_num_parameters_compression(model_5percent,model_reg))

keras_file = 'brokenLeNet.h5'
keras.models.save_model(model_5percent, keras_file, include_optimizer=False)
print('Saved broken LeNet Keras model to:', keras_file)

# %%
EPOCHS = 50
rseed = random.randint(1,1000)
try:
  result = model_5percent.fit(
      train_datagen.flow(X_train, y_train, batch_size = 64,seed=rseed,shuffle=True),
      validation_data=valid_datagen.flow(X_test, y_test, batch_size = 64,seed=rseed,shuffle=True),
      epochs = EPOCHS,
      verbose = 2,
      validation_freq = 5
  )
except KeyboardInterrupt:
  print("\n\nParalizado!\n\n")
#%%
ev_1_loss, ev_1_acc = model_1percent.evaluate(X_test,y_test)
ev_5_loss, ev_5_acc = model_5percent.evaluate(X_test,y_test)
ev_reg_loss, ev_reg_acc = model_reg.evaluate(X_test,y_test)
# %%

print(f"Model 1 Retraining Recovered { (ev_1_acc - ev_reg_acc +0.01) * 100}")
print(f"Model 5 Retraining Recovered { (ev_5_acc - ev_reg_acc +0.05) * 100}")

# %%
#Train 20percent error model
for i in range(len(acc)):
   if acc[i] > acc_full_model - 0.2:
      break
print(i+1)
model_20percent = LeNet_NN_broken(i+1)
set_model_weights(US[:,:i+1],VT[:i+1,:],weights,biases,model_20percent)
print(calc_num_parameters_compression(model_20percent,model_reg))

# %%
EPOCHS = 50
rseed = random.randint(1,1000)
try:
  result = model_20percent.fit(
      train_datagen.flow(X_train, y_train, batch_size = 64,seed=rseed,shuffle=True),
      validation_data=valid_datagen.flow(X_test, y_test, batch_size = 64,seed=rseed,shuffle=True),
      epochs = EPOCHS,
      verbose = 2,
      validation_freq = 5
  )
except KeyboardInterrupt:
  print("\n\nParalizado!\n\n")
#%%
ev_20_loss, ev_20_acc = model_20percent.evaluate(X_test,y_test)
print(f"Model 20 Retraining Recovered { (ev_20_acc - ev_reg_acc +0.2) * 100}")

#%%
#Train 35 percent error model
for i in range(len(acc)):
   if acc[i] > acc_full_model - 0.35:
      break
print(i+1)
model_35percent = LeNet_NN_broken(i+1)
set_model_weights(US[:,:i+1],VT[:i+1,:],weights,biases,model_35percent)
print(calc_num_parameters_compression(model_35percent,model_reg))

# %%
EPOCHS = 50
rseed = random.randint(1,1000)
try:
  result = model_35percent.fit(
      train_datagen.flow(X_train, y_train, batch_size = 64,seed=rseed,shuffle=True),
      validation_data=valid_datagen.flow(X_test, y_test, batch_size = 64,seed=rseed,shuffle=True),
      epochs = EPOCHS,
      verbose = 2,
      validation_freq = 5
  )
except KeyboardInterrupt:
  print("\n\nParalizado!\n\n")
#%%
ev_35_loss, ev_35_acc = model_35percent.evaluate(X_test,y_test)
print(f"Model 35 Retraining Recovered { (ev_35_acc - ev_reg_acc +0.35) * 100}")


# %%
#Train 50percent error model
for i in range(len(acc)):
   if acc[i] > acc_full_model - 0.5:
      break
print(i+1)
model_50percent = LeNet_NN_broken(i+1)
set_model_weights(US[:,:i+1],VT[:i+1,:],weights,biases,model_50percent)
print(calc_num_parameters_compression(model_50percent,model_reg))

# %%
EPOCHS = 50
rseed = random.randint(1,1000)
try:
  result = model_50percent.fit(
      train_datagen.flow(X_train, y_train, batch_size = 64,seed=rseed,shuffle=True),
      validation_data=valid_datagen.flow(X_test, y_test, batch_size = 64,seed=rseed,shuffle=True),
      epochs = EPOCHS,
      verbose = 2,
      validation_freq = 5
  )
except KeyboardInterrupt:
  print("\n\nParalizado!\n\n")
#%%
ev_50_loss, ev_50_acc = model_50percent.evaluate(X_test,y_test)
print(f"Model 50 Retraining Recovered { (ev_50_acc - ev_reg_acc +0.5) * 100}")

# %%
import matplotlib.pyplot as plt

# Your data
retrain_graph_x = [1, 5, 20, 35, 50]
retrain_graph_y = [
    (ev_1_acc - ev_reg_acc + 0.01) * 100,
    (ev_5_acc - ev_reg_acc + 0.05) * 100,
    (ev_20_acc - ev_reg_acc + 0.2) * 100,
    (ev_35_acc - ev_reg_acc + 0.35) * 100,
    (ev_50_acc - ev_reg_acc + 0.5) * 100
]

# Create the primary axis
fig, ax1 = plt.subplots()

# Plot original curves
ax1.plot(retrain_graph_x, retrain_graph_y, label="Observed Reconstruction", color="tab:blue")
ax1.plot(retrain_graph_x, retrain_graph_x, label="Perfect Reconstruction", color="tab:green")
ax1.set_xlabel("Original Compressed Model Error")
ax1.set_ylabel("Error")
ax1.legend(loc="upper left")

# Create secondary y-axis
ax2 = ax1.twinx()

# Compute and plot difference
difference = [y - x for x, y in zip(retrain_graph_x, retrain_graph_y)]
ax2.plot(retrain_graph_x, difference, label="Difference", color="tab:red", linestyle="--")
ax2.set_ylabel("Difference (Observed - Perfect)")
ax2.legend(loc="upper right")

plt.title("Observed vs Perfect Reconstruction and Their Difference")
plt.show()
# %%
