from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras.backend import int_shape
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import Embedding
from keras.layers import Reshape
from keras.layers import Conv1D
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import Add

import tensorflow as tf
import functools

def _scope(function):
  '''Successive calls to `function` will be variable-scoped
  with non-conflicting names based on `function.__name__`
  '''
  @functools.wraps(function)
  def wrapped(*args, **kwargs):
    with tf.variable_scope(None,
        default_name=function.__name__):
      return function(*args, **kwargs)
  return wrapped

def variable_scoped(cls):
  '''Wraps all non-hidden methods of `cls` with the
  `_scope` decorator
  '''
  for attr in dir(cls):
    method = getattr(cls, attr)
    if callable(method) and not attr.startswith('_'):
      setattr(cls, attr, _scope(method))
  return cls

@variable_scoped
class WaveNet(tuple):
  '''Class that creates a WaveNet defined by its input and output
  '''
  @staticmethod
  def CausalResidual(x, x0):
    '''Creates and applies a residual connection between tensors
    `x` and `x0` where `x0` is cropped to account for any "valid"
    (non-padded) convolutions applied to `x`
    '''
    def crop(inputs):
      x, x0 = inputs
      length_x = int_shape(x)[1] or tf.shape(x)[1]
      return x0[:, -length_x:]
    x0 = Lambda(crop)([x, x0])
    extra = int_shape(x)[-1] - int_shape(x0)[-1]
    if extra:
      extra = Conv1D(extra, 1)(x0)
      x0 = Concatenate()([x0, extra])
    return Add()([x, x0])

  @staticmethod
  def CausalLayer(x, width, dilation):
    '''Creates and applies a causal convolutional layer to tensor `x`
    with `width`-number of output units and some `dilation` rate
    '''
    x0 = x
    x = Activation('relu')(x)
    x = Conv1D(width // 2, 1)(x)
    x = Activation('relu')(x)
    x = Conv1D(width // 2, 2, dilation_rate=dilation)(x)
    x = Activation('relu')(x)
    x = Conv1D(width, 1)(x)
    return WaveNet.CausalResidual(x, x0)

  @staticmethod
  def CausalBlock(x, width, depth, dilation=True):
    '''Creates and applies a stack of causal convolutional layers
    with optional exponentially increasing dilation rate to tensor `x`
    '''
    for layer in range(depth):
      dilation = 2 ** layer if dilation else 1
      x = WaveNet.CausalLayer(x, width, dilation)
    return x

  @staticmethod
  def CausalSkip(tensors):
    '''Given a list of tensors, causally crops them to their smallest
    member's length, and concatenates them along their last axis
    '''
    def crop(inputs):
      min_length = tf.reduce_min(
        [tf.shape(t)[1] for t in inputs])
      return [t[:, -min_length:] for t in inputs]
    tensors = Lambda(crop)(tensors)
    return Concatenate()(tensors)

  def __new__(
      cls,
      channel_multiplier,
      blocks,
      layers_per_block,
      quantization,
      dilation,
    ):
    # allow variable-length stereo inputs
    inp = Input((None, 2))

    # embed categorical variables to a dense vector space
    x = Embedding(quantization, channel_multiplier // 2)(inp)

    # move stereo channels to the canonical channels axis
    x = Reshape((-1, channel_multiplier))(x)

    # apply a sequence of causal blocks, and cache the intermediate results
    skip = [x]
    for block in range(blocks):
      width = channel_multiplier * 2 ** block
      x = cls.CausalBlock(x, width, layers_per_block, dilation)
      skip.append(x)

    # concatenate all of the intermediate results
    x = cls.CausalSkip(skip)

    # final layers: back to categorical variable
    x = Activation('relu')(x)
    x = Conv1D(2 * quantization, 1)(x)
    x = Activation('relu')(x)
    x = Conv1D(2 * quantization, 1)(x)

    # move stereo channels back to penultimate axis
    out = Reshape((-1, 2, quantization))(x)

    # return a tuple of input and output
    return tuple.__new__(WaveNet, (inp, out))
