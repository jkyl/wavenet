from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras.backend import int_shape
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import Embedding
from keras.layers import Multiply
from keras.layers import Reshape
from keras.layers import Conv1D
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import Add
from keras import Model

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
    length_x = int_shape(x)[1] or tf.shape(x)[1]
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

@_variable_scope
def CausalSkip(tensors, receptive_field):
  def crop(inputs):
    return [t[:, receptive_field-1:] for t in inputs]
  tensors = Lambda(crop)(tensors)
  return Concatenate()(tensors)

def quantize(x, q):
  compressed = np.sign(x) * np.log(1 + (q - 1) * np.abs(x)) / np.log(q)
  bins = np.linspace(-1, 1, q + 1)
  quantized = np.digitize(compressed, bins)
  return quantized.astype(np.int32) - 1

def dequantize(x, q):
  bins = np.linspace(-1, 1, q + 1)
  centers = (bins[1:] + bins[:-1]) / 2.
  centers = np.sign(centers) * (1 / (q - 1)) * (q ** np.abs(centers) - 1) 
  return centers[x]

class WaveNet(Model):
  _sampling_rate = 44100 # Hz
  _quantization = 256 # 8-bit
   
  def __init__(self, channel_multiplier=32, blocks=5, octaves_per_block=13):
    self.receptive_field = blocks * 2 ** octaves_per_block - (blocks - 1) 
    inp = Input((self.receptive_field, 2))
    x = Embedding(self._quantization, channel_multiplier // 2)(inp)
    x = Reshape((-1, channel_multiplier))(x)
    skip = [x]
    for block in range(blocks):
      x = CausalBlock(x, channel_multiplier * 2 ** block, octaves_per_block)
      skip.append(x)
    x = CausalSkip(skip, self.receptive_field)
    x = Activation('relu')(x)
    x = Conv1D(2 * self._quantization, 1)(x)
    out = Reshape((-1, 2, self._quantization))(x) 
    super(WaveNet, self).__init__(inp, out)
    self.summary()
    print('Receptive field:', self.receptive_field)

  def get_data(self, data_dir, batch_size=3):
    data = []
    for mp3_file in glob.glob(os.path.join(data_dir, '*.mp3')):
      segment = pydub.AudioSegment.from_mp3(mp3_file)
      array = np.stack([
        np.frombuffer(channel._data, dtype=np.int16)
        for channel in segment.split_to_mono()], axis=1)
      data.append(array)
    padding = np.zeros((self.receptive_field, 2), dtype=np.float16)
    data = [data[i//2] if i % 2 else padding 
            for i in range(2 * len(data) + 1)]
    data = np.concatenate(data, axis=0)
    data = data.astype(np.float32) / 32768.
    data = quantize(data, self._quantization)
    def sample():
      while True:
        index = np.random.randint(data.shape[0] - self.receptive_field - 1)
        chunk = data[index:index+self.receptive_field+1]
        yield chunk[:-1], chunk[-1:]
    ds = tf.data.Dataset.from_generator(
      sample, (tf.int32, tf.int32), 
      ((self.receptive_field, 2), (1, 2)))
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
    learning_rate = 1e-3 * 2. ** -tf.cast(step // 50000, tf.float32)
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

  def generate(self, length_secs=0.1, weights_file=None):
    import tqdm
    if weights_file:
      self.load_weights(weights_file)
    length = int(length_secs * self._sampling_rate)
    data = quantize(0.001 * np.random.normal(
      size=(1, self.receptive_field + length, 2)
        ), self._quantization)
    for i in tqdm.trange(length):
      j = self.receptive_field + i
      input_buffer = data[:, i:j]
      output_logits = self.predict(input_buffer)
      output_sample = np.argmax(output_logits, axis=-1)
      data[:, j] = output_sample
    waveform = dequantize(data[0], self._quantization)
    np.save('waveform.npy', waveform)

if __name__ == '__main__': 
  WaveNet().train(
    '/Volumes/4TB/itunes/Josquin des Prez/Missa Pange lingua - Missa La sol fa re mi/',
    '/Volumes/4TB/training/wavenet/josquin/ch64_decay50k_rf-chunks/',
  )

