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
import time
import os

def _variable_scope(function):
  @functools.wraps(function)
  def wrapped(*args, **kwargs):
    with tf.variable_scope(None, 
        default_name=function.__name__):
      return function(*args, **kwargs)
  return wrapped

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

@_variable_scope
def CausalLayer(x, width, dilation):
  x0 = x
  x = Activation('relu')(x)
  x = Conv1D(width, 2, dilation_rate=dilation)(x)
  filt, gate = Lambda(lambda x: [x[..., :width//2], x[..., width//2:]])(x)
  x = Multiply()([Activation('tanh')(filt), Activation('sigmoid')(gate)])
  x = Conv1D(width, 1)(x)
  return CausalResidual(x, x0)

@_variable_scope
def CausalBlock(x, width, octaves): 
  for octave in range(octaves):
    dilation = 2 ** octave
    x = CausalLayer(x, width, dilation)
  return x

def quantize(x, q):
  companded = np.sign(x) * np.log(1 + (q - 1) * np.abs(x)) / np.log(q)
  bins = np.linspace(-1, 1, q + 1)
  quantized = np.digitize(companded, bins).astype(np.int32) - 1
  return quantized

class WaveNet(Model):
  _sampling_rate = 44100 # Hz
  _quantization = 256 # 8-bit
   
  def __init__(self, blocks=5, octaves_per_block=13):
    inp = Input((None, 2)) 
    x = Embedding(self._quantization, 8)(inp)
    x = Reshape((-1, 16))(x)
    for block in range(blocks):
      x = CausalBlock(x, 16 * 2 ** block, octaves_per_block)
    x = Activation('relu')(x)
    x = Conv1D(2 * self._quantization, 1)(x)
    out = Reshape((-1, 2, self._quantization))(x) 
    super(WaveNet, self).__init__(inp, out)
    self.summary()
    self.receptive_field = blocks * 2 ** octaves_per_block - (blocks - 1)
    print('Receptive field:', self.receptive_field)

  def get_data(self, data_dir, batch_size=4, length_secs=1):
    data = []
    for mp3_file in glob.glob(os.path.join(data_dir, '*.mp3')):
      segment = pydub.AudioSegment.from_mp3(mp3_file)
      array = np.stack([
        np.frombuffer(channel._data, dtype=np.int16)
        for channel in segment.split_to_mono()], axis=1)
      array = array.astype(np.float32) 
      array /= np.iinfo(np.int16).max
      data.append(array)
    padding = np.zeros((self.receptive_field, 2), dtype=np.float32)
    data = [data[i//2] if i % 2 else padding 
            for i in range(2 * len(data) + 1)]
    data = np.concatenate(data, axis=0)
    data = quantize(data, self._quantization)
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

  def fancy_save(self, model_dir, iteration=0):
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
    self.save(os.path.join(model_dir,
      'ckpt_iter-{}.h5'.format(str(iteration).zfill(10))))

  def train(self, data_dir, model_dir):
    features, labels = self.get_data(data_dir) 
    logits = self(features)
    loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels))
    step = tf.train.create_global_step()
    learning_rate = 1e-3 * 2. ** -tf.cast(step // 10000, tf.float32)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, step, self.trainable_weights)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      self.fancy_save(model_dir)
      tf.get_default_graph().finalize() 
      iter_ = 0
      loss_ = []
      time_ = time.time()
      while True:
        loss_ += sess.run([train_op, loss])[1:]
        iter_ += 1
        if not iter_ % 100:
          print('iter:', iter_, '\tloss:', np.mean(loss_))
          loss_ = []
        now = time.time()
        if now - time_ > 3600: # every hour
          print('saving checkpoint at iter:', iter_)
          self.fancy_save(model_dir, iter_)
          time_ = now
          
if __name__ == '__main__':
  WaveNet().train(
    '/Volumes/4TB/itunes/Josquin des Prez/Missa Pange lingua - Missa La sol fa re mi/',
    './checkpoints'
  )

