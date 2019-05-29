from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from wavenet import variable_scoped
from wavenet import WaveNet

import tensorflow as tf
import numpy as np
import argparse
import pydub
import keras
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
  bins = np.linspace(-1, 1, q + 1)
  centers = (bins[1:] + bins[:-1]) / 2.
  x = centers[x]
  x = np.sign(x) * (1 / (q - 1)) * (q ** np.abs(x) - 1)
  return x

@variable_scoped
class Trainer(keras.Model):
  _sampling_rate = 44100 # Hz
  _quantization = 256 # 8-bit

  def __init__(self, channel_multiplier, blocks, layers_per_block, **unused):
    '''Constructs a WaveNet model for training
    '''
    # compute the receptive field
    self.receptive_field = blocks * 2 ** layers_per_block - (blocks - 1)

    # get the default graph
    self.graph = tf.get_default_graph()

    # construct the model in the graph
    with self.graph.as_default():
      super(Trainer, self).__init__(*WaveNet(
        channel_multiplier=channel_multiplier,
        blocks=blocks,
        layers_per_block=layers_per_block,
        quantization=self._quantization,
        dilation=True,
      ))
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
    for mp3_file in sorted(glob.glob(os.path.join(data_dir, '*.mp3'))):
      segment = pydub.AudioSegment.from_mp3(mp3_file)
      array = np.stack([
        np.frombuffer(channel.raw_data, dtype=np.int16)
        for channel in segment.split_to_mono()], axis=1)
      if array.shape[-1] == 1:
        array = np.tile(array, (1, 2))
      elif array.shape[-1] != 2:
        raise ValueError(
          'Only mono and stereo audio supported (got {} channels)'
            .format(array.shape[-1])
      data.append(array)

    # zero-pad 1 RF at both ends of the dataset
    padding = np.zeros((self.receptive_field, 2), dtype=np.float16)
    data = [padding] + data + [padding]
    
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

    # construct a Dataset object from the generator
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
    learning_rate = 1e-3 * 2. ** -tf.cast(step // 250000, tf.float32)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(
      loss=loss, global_step=step, var_list=self.trainable_weights)

    # create some tensorboard summaries
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('learning_rate', learning_rate)
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(model_dir, self.graph)

    # start the session
    with tf.Session() as sess:

      # initialize and finalize the graph
      sess.run(tf.global_variables_initializer())
      self.fancy_save(model_dir)
      self.graph.finalize()

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
          writer.add_summary(sess.run(summary), iter_)
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
  return vars(p.parse_args())

if __name__ == '__main__':
  kwargs = parse_arguments()
  Trainer(**kwargs).train(**kwargs)
