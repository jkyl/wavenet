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
import argparse
import pydub
import glob
import time
import os

def quantize(x, q):
  '''Quantizes a signal `x` bound by the range [-1, 1] to the
  specified number of bins `q`, first applying a mu-law transformation
  '''
  compressed = np.sign(x) * np.log(1 + (q - 1) * np.abs(x)) / np.log(q)
  bins = np.linspace(-1, 1, q + 1)
  quantized = np.digitize(compressed, bins)
  return quantized.astype(np.int32) - 1

def dequantize(x, q):
  '''Un-bins a quantized signal `x` to a continuous one,
  then applyies an inverse mu-law transformation
  '''
  # TODO: double check impl
  bins = np.linspace(-1, 1, q + 1)
  centers = (bins[1:] + bins[:-1]) / 2.
  x = centers[x]
  x = np.sign(x) * (1 / (q - 1)) * (q ** np.abs(x) - 1)
  return x

def _variable_scope(function):
  '''Successive calls to `function` will be variable-scoped
  with non-conflicting names based on `function.__name__`
  '''
  @functools.wraps(function)
  def wrapped(*args, **kwargs):
    with tf.variable_scope(None,
        default_name=function.__name__):
      return function(*args, **kwargs)
  return wrapped

@_variable_scope
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

@_variable_scope
def CausalLayer(x, width, dilation):
  '''Creates and applies a causal convolutional layer to tensor `x`
  with `width`-number of hidden units and some `dilation` rate
  '''
  x0 = x
  x = Activation('relu')(x)
  x = Conv1D(width, 2, dilation_rate=dilation)(x)
  filt, gate = Lambda(lambda x: [x[..., :width//2], x[..., width//2:]])(x)
  x = Multiply()([Activation('tanh')(filt), Activation('sigmoid')(gate)])
  x = Conv1D(width, 1)(x)
  return CausalResidual(x, x0)

@_variable_scope
def CausalBlock(x, width, octaves):
  '''Creates and applies a stack of causal layers with exponentially
  increasing dilation rate (up to `octaves` powers of 2) to tensor `x`
  '''
  for octave in range(octaves):
    dilation = 2 ** octave
    x = CausalLayer(x, width, dilation)
  return x

@_variable_scope
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

class WaveNet(Model):
  _sampling_rate = 44100 # Hz
  _quantization = 256 # 8-bit

  def __init__(self, channel_multiplier, blocks, octaves_per_block, **unused):
    '''Constructs a WaveNet model
    '''
    # compute the receptive field
    self.receptive_field = blocks * 2 ** octaves_per_block - (blocks - 1)

    # allow variable-length stereo inputs
    inp = Input((None, 2))

    # embed categorical variables to a dense vector space
    x = Embedding(self._quantization, channel_multiplier // 2)(inp)

    # move stereo channels to the canonical channels axis
    x = Reshape((-1, channel_multiplier))(x)

    # apply a sequence of causal blocks, and cache the intermediate results
    skip = [x]
    for block in range(blocks):
      x = CausalBlock(x, channel_multiplier * 2 ** block, octaves_per_block)
      skip.append(x)

    # concatenate all of the intermediate results
    x = CausalSkip(skip)

    # project back to a categorical variable
    x = Activation('relu')(x)
    x = Conv1D(2 * self._quantization, 1)(x)

    # move stereo channels back to penultimate axis
    out = Reshape((-1, 2, self._quantization))(x)

    # call the parent class's constructor
    super(WaveNet, self).__init__(inp, out)

    # log some model info
    self.summary()
    print('Receptive field:', self.receptive_field)

  def get_data(self, data_dir, length_secs, batch_size):
    '''Loads .mp3 files from `data_dir`, preprocesses, and
    yields batches of random chunks with a tf.data.Dataset
    '''
    # compute the length in samples of one training example
    length = int(length_secs * self._sampling_rate)
    assert length >= self.receptive_field, length

    # grab all of the .mp3 data
    data = []
    for mp3_file in glob.glob(os.path.join(data_dir, '*.mp3')):
      segment = pydub.AudioSegment.from_mp3(mp3_file)
      array = np.stack([
        np.frombuffer(channel._data, dtype=np.int16)
        for channel in segment.split_to_mono()], axis=1)
      data.append(array)

    # zero-pad between tracks and at both ends of the album
    padding = np.zeros((self.receptive_field, 2), dtype=np.float16)
    data = [data[i//2] if i%2 else padding for i in range(2 * len(data) + 1)]

    # merge it all together as a [-1, 1] bounded float array
    data = np.concatenate(data, axis=0)
    data = data.astype(np.float32) / 32768.

    # quantize the entire array (represent as categorical)
    data = quantize(data, self._quantization)

    # define a generator that yields samples from the array
    def sample():
      while True:
        index = np.random.randint(data.shape[0]-length-1)
        chunk = data[index:index+length+1]
        yield chunk[:-1], chunk[self.receptive_field:]

    # construct a tf.data.Dataset object from the generator
    ds = tf.data.Dataset.from_generator(
      generator=sample,
      output_types=(tf.int32, tf.int32),
      output_shapes=((length, 2), (length - self.receptive_field + 1, 2)),
    )
    # batch the samples and prefetch
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)

    # return symbolic tensors
    iterator = ds.make_one_shot_iterator()
    return iterator.get_next()

  def fancy_save(self, model_dir, iteration=0):
    '''Saves a formatted model checkpoint, creating a directory if necessary
    '''
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
    self.save(os.path.join(model_dir,
      'ckpt_iter-{}.h5'.format(str(iteration).zfill(10))))

  def train(self, data_dir, model_dir, batch_size, length_secs, **unused):
    '''Trains the WaveNet model on .mp3 files
    '''
    # get the training data tensors
    features, labels = self.get_data(
      data_dir=data_dir,
      batch_size=batch_size,
      length_secs=length_secs,
    )
    # make the predictions
    logits = self(features)

    # loss is cross-entropy with the true t+1 data
    loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels))

    # use default Adam with learning rate decay
    step = tf.train.create_global_step()
    learning_rate = 1e-3 * 2. ** -tf.cast(step // 50000, tf.float32)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, step, self.trainable_weights)

    # start the session
    with tf.Session() as sess:

      # initialize and finalize the graph
      sess.run(tf.global_variables_initializer())
      self.fancy_save(model_dir)
      tf.get_default_graph().finalize()

      # initialize loop variables
      iter_ = 0
      loss_ = []
      time_ = time.time()

      # training loop
      while True:
        loss_ += sess.run([train_op, loss])[1:]
        iter_ += 1

        # log the average loss for the last 100 steps
        if not iter_ % 100:
          print('iter:', iter_, '\tloss:', np.mean(loss_))
          loss_ = []

        # save checkpoints at regular time intervals
        now = time.time()
        if now - time_ > 3600: # every hour
          print('saving checkpoint at iter:', iter_)
          self.fancy_save(model_dir, iter_)
          time_ = now

def parse_arguments():
  p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  p.add_argument(
    'data_dir',
    type=str,
    help='directory containing .mp3 files to use as training data',
  )
  p.add_argument(
    'model_dir',
    type=str,
    help='directory in which to save checkpoints and summaries',
  )
  p.add_argument(
    '-ch',
    dest='channel_multiplier',
    type=int,
    default=32,
    help='multiplicative factor for all hidden units',
  )
  p.add_argument(
    '-bk',
    dest='blocks',
    type=int,
    default=5,
    help='number of causal blocks in the network',
  )
  p.add_argument(
    '-oc',
    dest='octaves_per_block',
    type=int,
    default=13,
    help='number of dilated convolutions per causal block',
  )
  p.add_argument(
    '-bs',
    dest='batch_size',
    type=int,
    default=1,
    help='number of training examples per parameter update',
  )
  p.add_argument(
    '-ls',
    dest='length_secs',
    type=float,
    default=1.,
    help='length in seconds of a single training example',
  )
  return vars(p.parse_args())

if __name__ == '__main__':
  kwargs = parse_arguments()
  model = WaveNet(**kwargs)
  model.train(**kwargs)
