from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from wavenet import WaveNet

import tensorflow as tf
import numpy as np
import argparse
import pydub
import keras
import glob
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
  bins = np.linspace(-1, 1, q + 1)
  centers = (bins[1:] + bins[:-1]) / 2.
  x = centers[x]
  x = np.sign(x) * (1 / (q - 1)) * (q ** np.abs(x) - 1)
  return x

def train(features, labels, mode, params):
  '''Constructs a WaveNet model for training
  '''
  # build the keras model
  model = keras.Model(*WaveNet(**params))

  # loss is cross-entropy with the true t+1 data
  loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=model(features), labels=labels))

  # use default Adam with learning rate decay
  step = tf.train.get_global_step()
  learning_rate = 1e-3 * 2. ** -tf.cast(step // params['decay'], tf.float32)
  train_op = tf.train.AdamOptimizer(learning_rate).minimize(
    loss=loss, global_step=step, var_list=model.trainable_weights)

  # create some tensorboard summaries
  tf.summary.scalar('loss', loss)
  tf.summary.scalar('learning_rate', learning_rate)

  # return an EstimatorSpec
  return tf.estimator.EstimatorSpec(
    mode=mode, loss=loss, train_op=train_op)

def get_data(params):
  '''Loads .mp3 files from `data_dir`, preprocesses, and
  yields batches of random chunks with a tf.data.Dataset
  '''
  # compute the receptive field
  receptive_field = WaveNet.get_receptive_field(
    params['blocks'], params['layers_per_block'])

  # compute the length in samples of one training example
  length = int(params['length_secs'] * params['sampling_rate'])
  assert length >= receptive_field, length

  # grab all of the .mp3 data
  data_dir_or_file = params['data_dir']
  if os.path.isfile(data_dir_or_file) and data_dir_or_file.endswith('.mp3'):
    files = [data_dir_or_file]
  elif os.path.isdir(data_dir_or_file):
    files = sorted(glob.glob(os.path.join(data_dir_or_file, '*.mp3')))
  else:
    raise OSError('{} does not exist'.format(data_dir_or_file))
  data = []
  for mp3_file in files:
    segment = pydub.AudioSegment.from_mp3(mp3_file)
    array = np.stack([
      np.frombuffer(channel.raw_data, dtype=np.int16)
      for channel in segment.split_to_mono()], axis=1)
    if array.shape[-1] == 1:
      array = np.tile(array, (1, 2))
    elif array.shape[-1] != 2:
      raise ValueError(
        'Only mono and stereo audio supported (got {} channels)'
          .format(array.shape[-1]))
    data.append(array)

  # zero-pad 1 RF at both ends of the dataset
  padding = np.zeros((receptive_field, 2), dtype=np.float16)
  data = [padding] + data + [padding]

  # merge it all together as a [-1, 1] bounded float array
  data = np.concatenate(data, axis=0)
  data = data.astype(np.float32) / 32768.

  # quantize the entire array (represent as categorical)
  data = quantize(data, params['quantization'])

  # define a generator that yields samples from the array
  def sample():
    while True:
      index = np.random.randint(data.shape[0]-length-1)
      chunk = data[index:index+length+1]
      yield chunk[:-1], chunk[receptive_field:]

  # construct a Dataset object from the generator
  ds = tf.data.Dataset.from_generator(
    generator=sample,
    output_types=(tf.int32, tf.int32),
    output_shapes=((length, 2), (length - receptive_field + 1, 2)),
  )
  # batch the samples and prefetch
  ds = ds.batch(params['batch_size'])
  ds = ds.prefetch(1)

  # return the whole dataset
  return ds

def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.estimator.Estimator(
    model_fn=train,
    params=vars(args),
    model_dir=args.model_dir,
    config=tf.estimator.RunConfig(
      save_checkpoints_secs=3600,
      save_summary_steps=10)
  ).train(get_data, steps=1000000)

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
    default=16,
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
    '-lp',
    dest='layers_per_block',
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
    default=2.,
    help='length in seconds of a single training example',
  )
  p.add_argument(
    '-sr',
    dest='sampling_rate',
    type=int,
    default=44100,
    help='number of audio samples per second',
  )
  p.add_argument(
    '-qz',
    dest='quantization',
    type=int,
    default=256,
    help='number of bins in which to quantize the audio signal',
  )
  p.add_argument(
    '-dy',
    dest='decay',
    type=int,
    default=100000,
    help='number of updates after which to halve the learning rate, successively',
  )
  return p.parse_args()

if __name__ == '__main__':
  main(parse_arguments())
