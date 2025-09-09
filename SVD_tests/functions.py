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

'''
Function that Sets loads model new model weights in based on rank 
decomposition for each layer

'''
def SVD_set_model_weights( model,orig_weights,biases,rank_cut):
    S_matrix = []
    r_weights = []

    # Make SVD Conversion for each layer
    U, S, VT = [],[],[] 
    for weight,r_cut in zip(orig_weights,rank_cut):
        t_U , t_S , t_VT = np.linalg.svd(weight)
        #Perform rank decomposition based on rank parameter
        if (r_cut < len(t_S)):
            U_r = np.delete(t_U,[range(len(t_S)-r_cut,len(t_S),1)], axis=1)
            VT_r  = np.delete(t_VT,[range(len(t_S)-r_cut,len(t_S),1)], axis=0)
            S_r = t_S if r_cut == 0 else t_S[:-r_cut]
            U.append(U_r)
            S.append(S_r)
            VT.append(VT_r)
        else:
            print(f"Rank selected is bigger then full :{rank_cut}")                
    #All SVDs already decomposed
    #Reconstruct weight from SVD
    for w,s,u,vt in zip(orig_weights,S,U,VT):
        S_matrix.append(np.zeros((u.shape[1],len(vt))))
        np.fill_diagonal(S_matrix[-1], s)
        #print(f"({u.shape = },{S_matrix[-1].shape = },{vt.shape = }",)
        r_weights.append(u @ S_matrix[-1] @ vt)
        #print("Error for each layer reconstructed:",sum(sum(r_weights[-1] - w)))

    weights_list = []
    for w , b in zip(r_weights,biases):
        weights_list.append(w)
        weights_list.append(b)

    model.set_weights(weights_list)

def load_mnist():
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

    return X_train, y_train, X_test, y_test


def calc_compression_per_layer(m,n,l):
  
    compression = []
    for k in range(n,0,-1):
        n_sum = (k * (m + (2*n -1)) + k*m)
        n_original= ( m*l*(2*n -1))
        compression.append(n_original/n_sum)

    return compression




def calc_num_parameters_compression(comp_model,orig_model):
    
    total_compression = orig_model.count_params()/comp_model.count_params()
    
    compressed_layer = orig_model.layers[1]
    num_params_layer = np.sum([np.prod(w.shape) for w in compressed_layer.trainable_weights])
    
    layer_1_0 = comp_model.layers[1]
    layer_1_1 = comp_model.layers[2]
    num_params_broken = np.sum([np.prod(w.shape) for w in layer_1_0.trainable_weights])
    
    num_params_broken += np.sum([np.prod(w.shape) for w in layer_1_1.trainable_weights])
    
    layer_compression = num_params_layer/num_params_broken
    
    print(f"Total Model Compression: {total_compression:.2f} ({100/total_compression:.2f}%)")
    print(f"Layer Compression: {layer_compression:.2f} ({100/layer_compression:.2f}%)")
    return total_compression, layer_compression
# %%
