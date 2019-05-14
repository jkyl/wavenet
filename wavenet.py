from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras import Model
from keras.layers import Add
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Conv1D
from keras.layers import Multiply
from keras.layers import Activation
from keras.layers import Concatenate
from keras.backend import int_shape

import tensorflow as tf
import numpy as np
import functools
import pydub
import glob
import os

_SAMPLING_RATE = 44100 # Hz

def _variable_scope(function):
  @functools.wraps(function)
  def wrapped(*args, **kwargs):
    with tf.variable_scope(None, 
        default_name=function.__name__):
      return function(*args, **kwargs)
  return wrapped

class WaveNet(Model):
  
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
      extra = Conv1D(extra, 1, use_bias=False)(x0)
      x0 = Concatenate()([x0, extra])
    return Add()([x, x0])

  @staticmethod
  @_variable_scope
  def CausalLayer(x, width, dilation):
    x0 = x
    x = Activation('relu')(x)
    x = Conv1D(width, 2, dilation_rate=dilation, padding='valid')(x)
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
  
  def __init__(self, blocks=3, octaves_per_block=14):
    inp = Input((None, 2))
    x = Conv1D(16, 1)(inp)
    for block in range(blocks):
      width = 16 * 2 ** block
      x = self.CausalBlock(x, width, octaves_per_block)
    x = Activation('relu')(x)
    x = Conv1D(2, 1)(x)
    out = Activation('tanh')(x)
    super(WaveNet, self).__init__(inp, out)
    self.receptive_field = blocks * 2 ** octaves_per_block - (blocks - 1)
    self.summary()
    print('receptive field:', self.receptive_field)

  def get_data(self, data_dir, batch_size=4, length_secs=2):
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
    length = _SAMPLING_RATE * length_secs
    def sample():
      while True:
        index = np.random.randint(data.shape[0] - length)
        chunk = data[index:index+length]
        yield chunk[:-1], chunk[self.receptive_field:]
    ds = tf.data.Dataset.from_generator(
      sample, (tf.float32, tf.float32), 
      ((length - 1, 2), (length - self.receptive_field, 2)))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(8)
    return ds

  def train(self, data_dir):
    data = self.get_data(data_dir)
    X_train, Y_train = data.make_one_shot_iterator().get_next()
    Y_pred = self(X_train)
    loss = tf.reduce_mean((Y_pred - Y_train)**2)
    psnr = 10 * (2 * tf.log(2.) - tf.log(loss)) / tf.log(10.)
    optimizer = tf.train.AdamOptimizer(2e-4)
    train_op = optimizer.minimize(loss, var_list=self.trainable_weights)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for i in range(1000):
        for j in range(100):
          print(sess.run([train_op, psnr])[-1])

if __name__ == '__main__':
  WaveNet().train('/Volumes/4TB/itunes/Josquin des Prez/Missa Pange lingua - Missa La sol fa re mi/')

