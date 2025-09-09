#%%S
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
from tensorflow.keras.models import load_model
lenet = load_model('/home/isi/Documents/Mestrado/SVD_tests/brokenLeNet.h5')
lenet.summary()

# %%
