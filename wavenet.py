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
import numpy as np
import functools
import argparse
import pydub
import glob
import sys
import os

def scope(function):
  '''Successive calls to `function` will be variable-scoped
  with non-conflicting names based on `function.__name__`
  '''
  @functools.wraps(function)
  def wrapped(*args, **kwargs):
    with tf.variable_scope(None,
        default_name=function.__name__):
      return function(*args, **kwargs)
  return wrapped

@scope
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

@scope
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
  return CausalResidual(x, x0)

@scope
def CausalBlock(x, width, depth):
  '''Creates and applies a stack of causal convolutional layers
  with optional exponentially increasing dilation rate to tensor `x`
  '''
  for layer in range(depth):
    dilation = 2 ** layer
    x = CausalLayer(x, width, dilation)
  return x

@scope
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

def get_receptive_field(blocks, layers_per_block):
  '''Computes the receptive field of a WaveNet model
  '''
  return blocks * 2 ** layers_per_block - (blocks - 1)

def forward_pass(
    input_tensor, 
    channel_multiplier,
    blocks,
    layers_per_block,
    quantization, 
    mode,
    **unused,
  ):
  '''Creates and applies the WaveNet model to an input tensor
  '''
  # allow variable-length stereo inputs
  inp = Input((None, 2), tensor=input_tensor)

  # embed categorical variables to a dense vector space
  x = Embedding(quantization, channel_multiplier // 2)(inp)

  # move stereo channels to the canonical channels axis
  x = Reshape((-1, channel_multiplier))(x)

  # apply a sequence of causal blocks, and cache the intermediate results
  skip = [x]
  for block in range(blocks):
    width = channel_multiplier * 2 ** block
    x = CausalBlock(x, width, layers_per_block)
    skip.append(x)

  # concatenate all of the intermediate results
  x = CausalSkip(skip)

  # final layers: back to categorical variable
  x = Activation('relu')(x)
  x = Conv1D(2 * quantization, 1)(x)

  # move stereo channels back to penultimate axis
  out = Reshape((-1, 2, quantization))(x)
  
  # apply softmax layer to predictions
  if mode == tf.estimator.ModeKeys.PREDICT:
    out = Activation('softmax')(out)

  # return the output tensor
  return out

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

def main(args):
  '''Entrypoint for wavenet.py
  '''  
  def model_fn(features, labels, mode):
    '''Returns a WaveNet EstimatorSpec for training or prediction
    '''
    # train mode:
    if mode == tf.estimator.ModeKeys.TRAIN:

      # build and apply the model to the input features
      logits = forward_pass(features, **vars(args))
      
      # loss is cross-entropy with the true t+1 data
      loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels))

      # use default Adam with learning rate decay
      step = tf.train.get_global_step()
      learning_rate = 1e-3 * 2. ** -tf.cast(step // args.decay, tf.float32)
      train_op = tf.train.AdamOptimizer(learning_rate).minimize(
        loss=loss, global_step=step)

      # create some tensorboard summaries
      tf.summary.scalar('loss', loss)
      tf.summary.scalar('learning_rate', learning_rate)

      # define the EstimatorSpec
      return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op)

    # predict mode
    elif mode == tf.estimator.ModeKeys.PREDICT:
      
      # build and apply the model to the input features
      logits = forward_pass(features, **vars(args))

      # get the predictions
      predictions = tf.argmax(logits, axis=-1)

      return tf.estimator.EstimatorSpec(
        mode=mode, predictions=predictions)
 
  def input_fn(mode):
    '''Returns a tf.data.Dataset for training or prediction
    '''
    # compute the receptive field
    receptive_field = get_receptive_field(
      args.blocks, args.layers_per_block)
    
    # train mode:
    if mode == tf.estimator.ModeKeys.TRAIN:    
      
      # compute the length in samples of one training example
      length = int(args.length_secs * 44100)
      assert length >= receptive_field, length

      # grab all of the .mp3 data
      if os.path.isfile(args.data_dir) and args.data_dir.endswith('.mp3'):
        files = [args.data_dir]
      elif os.path.isdir(args.data_dir):
        files = sorted(glob.glob(os.path.join(args.data_dir, '*.mp3')))
      else:
        raise OSError('{} does not exist'.format(args.data_dir))
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
      data = quantize(data, args.quantization)

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
      ds = ds.batch(args.batch_size)
      ds = ds.prefetch(1)
      return ds

    # predict mode:
    elif mode == tf.estimator.ModeKeys.PREDICT:
      return 128 * tf.ones((1, receptive_field, 2), dtype=tf.int32)
       
  # enable log messages
  tf.logging.set_verbosity(tf.logging.INFO)

  # create the estimator
  estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=args.model_dir,
  )
  # either train or predict
  if args.mode == 'train':
    estimator.train(input_fn, steps=1000000)
  elif args.mode == 'predict':
    for yhat in estimator.predict(input_fn):
      print(yhat)

def parse_arguments():
  '''Parses command line arguments to wavenet.py
  '''
  p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  sp = p.add_subparsers(dest='mode')
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
    '-qz',
    dest='quantization',
    type=int,
    default=256,
    help='number of bins in which to quantize the audio signal',
  )
  train = sp.add_parser(
    'train',
    help='train a WaveNet model',
  )
  train.add_argument(
    'data_dir',
    type=str,
    help='directory containing .mp3 files to use as training data',
  )
  train.add_argument(
    '-dy',
    dest='decay',
    type=int,
    default=100000,
    help='number of updates after which to halve the learning rate',
  )
  predict = sp.add_parser(
    'predict',
    help='generate audio with a pre-trained WaveNet model'
  )
  return p.parse_args()

if __name__ == '__main__':
  main(parse_arguments())
