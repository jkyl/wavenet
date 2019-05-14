from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.backend import int_shape
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
import numpy as np
import functools
import pydub

def _scoped_staticmethod(function):
  @functools.wraps(function)
  def wrapped(*args, **kwargs):
    with tf.variable_scope(None, 
        default_name=function.__name__):
      return function(*args, **kwargs)
  return staticmethod(wrapped)

class WaveNet(Model):
  def __init__(self, blocks=2, octaves_per_block=14):
    inp = Input((None, 2))
    x = Conv1D(32, 1)(inp)
    for block in range(blocks):
      width = 32 * 2 ** block
      x = self.CausalBlock(x, width, octaves_per_block)
    x = Activation('relu')(x)
    x = Conv1D(2, 1)(x)
    out = Activation('tanh')(x)
    super(WaveNet, self).__init__(inp, out)
    self.receptive_field = blocks * np.sum(2 ** np.arange(octaves_per_block + 1)) - (blocks - 1)

  def get_data(self, mp3_file, batch_size=1, length=3*44100):
    segment = pydub.AudioSegment.from_mp3(mp3_file)
    array = np.stack([
      np.frombuffer(channel._data, dtype=np.int16)
      for channel in segment.split_to_mono()], axis=1)
    array = array.astype(np.float32) / np.iinfo(np.int16).max
    def sample():
      while True:
        index = np.random.randint(array.shape[0] - length)
        chunk = array[index:index+length]
        yield chunk[:-1], chunk[self.receptive_field:]
    ds = tf.data.Dataset.from_generator(
      sample, (tf.float32, tf.float32), 
      ((length - 1, 2), (length - self.receptive_field, 2)))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(8)
    return ds

  def train(self, mp3_file):
    self.compile(loss='mse', optimizer=Adam(3e-4))
    data = self.get_data(mp3_file)
    self.fit(data, steps_per_epoch=100, epochs=100)

  @_scoped_staticmethod
  def CausalResidual(x, x0):
    _, length_x, width_x = int_shape(x)
    _, length_x0, width_x0 = int_shape(x0)
    if length_x is None:
      length_x = tf.shape(x)[1]
    x0 = Lambda(lambda t: t[:, -length_x:])(x0)
    extra = width_x - width_x0
    if extra:
      extra = Conv1D(extra, 1, use_bias=False)(x0)
      x0 = Concatenate()([x0, extra])
    return Add()([x, x0])

  @_scoped_staticmethod
  def CausalLayer(x, width, dilation):
    x0 = x
    x = Activation('relu')(x)
    x = Conv1D(width, 3, dilation_rate=dilation)(x)
    filt, gate = Lambda(lambda x: (x[..., :width//2], x[..., width//2:]))(x)
    x = Multiply()([Activation('tanh')(filt), Activation('sigmoid')(gate)])
    x = Conv1D(width, 1)(x)
    return WaveNet.CausalResidual(x, x0)
 
  @_scoped_staticmethod
  def CausalBlock(x, width, octaves): 
    for octave in range(octaves):
      dilation = 2 ** octave
      x = WaveNet.CausalLayer(x, width, dilation)
    return x

if __name__ == '__main__':
  WaveNet().train('/Volumes/4TB/torrents/Holly Herndon - PROTO (2019) [WEB V0]/09 - SWIM.mp3')

