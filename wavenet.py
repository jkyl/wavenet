from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras import Model
from keras.layers import Add
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Conv1D
from keras.layers import Reshape
from keras.layers import Multiply
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import Concatenate
from keras.backend import int_shape

import tensorflow as tf
import numpy as np
import functools
import pydub
import glob
import os

def _variable_scope(function):
  @functools.wraps(function)
  def wrapped(*args, **kwargs):
    with tf.variable_scope(None, 
        default_name=function.__name__):
      return function(*args, **kwargs)
  return wrapped

class WaveNet(Model):
  _sampling_rate = 44100 # Hz
  
  @staticmethod
  @_variable_scope
  def CausalResidual(x, x0):
    def crop(inputs):
      x, x0 = inputs
      length_x = tf.shape(x)[1]
      return x0[:, -length_x:]
    x0 = Lambda(crop)([x, x0])
    extra = int_shape(x)[-1] - int_shape(x0)[-1]
    if extra:
      extra = Conv1D(extra, 1)(x0)
      x0 = Concatenate()([x0, extra])
    return Add()([x, x0])

  @staticmethod
  @_variable_scope
  def CausalLayer(x, width, dilation):
    x0 = x
    x = Activation('relu')(x)
    x = Conv1D(width, 2, dilation_rate=dilation)(x)
    filt, gate = Lambda(lambda x: [x[..., :width//2], x[..., width//2:]])(x)
    x = Multiply()([Activation('tanh')(filt), Activation('sigmoid')(gate)])
    x = Conv1D(width, 1)(x)
    return WaveNet.CausalResidual(x, x0)
  
  @staticmethod
  @_variable_scope
  def CausalBlock(x, width, octaves): 
    for octave in range(octaves):
      dilation = 2 ** octave
      x = WaveNet.CausalLayer(x, width, dilation)
    return x
  
  def __init__(self, blocks=4, octaves_per_block=13):
    inp = Input((None, 2))
    x = Embedding(256, 8)(inp)
    x = Reshape((-1, 16))(x)
    for block in range(blocks):
      width = 16 * 2 ** block
      x = self.CausalBlock(x, width, octaves_per_block)
    x = Activation('relu')(x)
    x = Conv1D(512, 1)(x)
    out = Reshape((-1, 2, 256))(x)
    super(WaveNet, self).__init__(inp, out)
    self.receptive_field = blocks * 2 ** octaves_per_block - (blocks - 1)
    self.summary()
    print('Receptive field:', self.receptive_field)

  @staticmethod
  def quantize(x, levels=256):
    companded = np.sign(x) * np.log(1 + (levels - 1) * np.abs(x)) / np.log(levels)
    bins = np.linspace(-1, 1, levels + 1)
    return np.digitize(companded, bins).astype(np.int32) - 1

  def get_data(self, data_dir, batch_size=8, length_secs=1):
    data = []
    for mp3_file in glob.glob(os.path.join(data_dir, '*.mp3')):
      segment = pydub.AudioSegment.from_mp3(mp3_file)
      array = np.stack([
        np.frombuffer(channel._data, dtype=np.int16)
        for channel in segment.split_to_mono()], axis=1)
      padding = np.zeros((self.receptive_field, 2), dtype=np.int16)
      array = np.concatenate([padding, array], axis=0)
      array = array.astype(np.float32) 
      array /= np.iinfo(np.int16).max
      data.append(array)
    data.append(padding.astype(np.float32))
    data = np.concatenate(data, axis=0)
    data = self.quantize(data)
    length = self._sampling_rate * length_secs
    def sample():
      while True:
        index = np.random.randint(data.shape[0] - length)
        chunk = data[index:index+length]
        yield chunk[:-1], chunk[self.receptive_field:]
    ds = tf.data.Dataset.from_generator(
      sample, (tf.int32, tf.int32), 
      ((length - 1, 2), (length - self.receptive_field, 2)))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(8)
    return ds.make_one_shot_iterator().get_next()

  def train(self, data_dir):
    X_train, Y_train = self.get_data(data_dir) 
    logits = self(X_train)
    loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=Y_train))
    optimizer = tf.train.AdamOptimizer(5e-4)
    train_op = optimizer.minimize(loss, var_list=self.trainable_weights)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for i in range(1000):
        for j in range(100):
          print(sess.run([train_op, loss])[1])

if __name__ == '__main__':
  WaveNet().train('/Volumes/4TB/itunes/Josquin des Prez/Missa Pange lingua - Missa La sol fa re mi/')

