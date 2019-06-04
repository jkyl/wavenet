from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Add
from tensorflow.keras import Model

import tensorflow as tf
import numpy as np
import functools
import argparse
import pydub
import glob
import json
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

def quantize(x, q):
  '''Quantizes a signal `x` bound by the range [-1, 1] to the
  specified number of bins `q`, first applying a mu-law companding
  transformation, following https://arxiv.org/abs/1609.03499,
  section 2.2
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

@scope
def CausalResidual(x, x0):
  '''Creates and applies a residual connection between tensors
  `x` and `x0` where `x0` is cropped to account for any "valid"
  (non-padded) convolutions applied to `x`
  '''
  def crop(inputs):
    x, x0 = inputs
    return x0[:, -(int_shape(x)[1] or tf.shape(x)[1]):]
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
  with exponentially increasing dilation rate to tensor `x`
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
    min_length = tf.reduce_min([tf.shape(t)[1] for t in inputs])
    return [t[:, -min_length:] for t in inputs]
  tensors = Lambda(crop)(tensors)
  return Concatenate()(tensors)

def get_receptive_field(blocks, layers_per_block):
  '''Computes the receptive field of a WaveNet model
  '''
  return blocks * 2 ** layers_per_block - (blocks - 1)

def save_output(output_filepath, stereo_array):
  '''Writes a stereo array to the output filepath --
  accepts .wav, .mp3, .ogg, or anything else supported by ffmpeg
  '''
  pydub.AudioSegment.from_mono_audiosegments(*[
    pydub.AudioSegment(
      (channel * 32768).astype(np.int16).tobytes(),
      frame_rate=44100,
      sample_width=np.dtype(np.int16).itemsize,
      channels=1,
    )
    for channel in stereo_array.reshape((-1, 2)).T
  ]).export(output_filepath, format=output_filepath.split('.')[-1])

def build_model(
    *,
    channel_multiplier,
    blocks,
    layers_per_block,
    quantization,
    **unused,
  ):
  '''Creates a WaveNet Keras model
  '''
  inp = Input((None, 2))

  # embed categorical variables to a dense vector space
  x = Embedding(quantization, channel_multiplier // 2)(inp)

  # move stereo channels to the canonical channels axis
  x = Reshape((-1, channel_multiplier))(x)

  # apply a sequence of causal blocks
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

  # return the output tensor
  return Model(inp, out)

def main(args):
  '''Entrypoint for wavenet.py
  '''
  if args.mode == 'predict':
    # get the configuration of the trained model
    with open(os.path.join(args.model_dir, 'config.json')) as f:
      config = json.loads(f.read())
      config.update(vars(args))
      args = argparse.Namespace(**config)

  # compute the receptive field
  receptive_field = get_receptive_field(
    args.blocks, args.layers_per_block)

  def input_fn(mode):
    '''Returns a tf.data.Dataset for training,
    or a seed tensor for generation
    '''
    # predict mode:
    if mode == tf.estimator.ModeKeys.PREDICT:

      # seed the generative model with a blank slate
      seed = np.zeros((1, receptive_field, 2), dtype=np.float32)

      # randomize the last sample
      seed[:, -1] = np.clip(np.random.normal(), -1, 1)

      # quantize
      seed = quantize(seed, args.quantization)

      # return as constant
      return tf.constant(seed)

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

  def model_fn(features, labels, mode):
    '''Returns a WaveNet EstimatorSpec for training or prediction
    '''
    # build the model
    model = build_model(**vars(args))

    # predict mode:
    if mode == tf.estimator.ModeKeys.PREDICT:

      # do autoregressive predictions
      _, predictions = tf.while_loop(
        lambda i, _: tf.less(i, int(args.length_secs * 44100)),
        lambda i, f: [i + 1,
          tf.concat([f, tf.argmax(model(f[:,-receptive_field:]),
            axis=-1, output_type=tf.int32)], axis=1)],
        [tf.constant(0), features],
        shape_invariants=[
          tf.TensorShape([]),
          tf.TensorShape([1, None, 2])]
      )
      # return an EstimatorSpec (predict mode)
      return tf.estimator.EstimatorSpec(
        mode=mode, predictions=predictions)

    # apply the model to the input features
    logits = model(features)

    # loss is cross-entropy with the true t+1 data
    loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels))

    # use default Adam with learning rate decay
    step = tf.train.get_global_step()
    learning_rate = 1e-3 * 2. ** -tf.cast(step // args.decay, tf.float32)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(
      loss=loss, global_step=step)

    # get an accuracy metric
    accuracy = tf.reduce_mean(
      tf.cast(tf.equal(tf.cast(tf.argmax(
        logits, axis=-1), tf.int32), labels), tf.float32))

    # create some tensorboard summaries
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('learning_rate', learning_rate)

    # save the command line args after model_dir is created
    class ConfigSaverHook(tf.train.SessionRunHook):
      def after_create_session(self, session, coord):
        with open(os.path.join(args.model_dir, 'config.json'), 'w') as f:
          f.write(json.dumps(vars(args)))

    # return an EstimatorSpec (train mode)
    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      training_hooks=[ConfigSaverHook()],
    )
  # enable log messages
  tf.logging.set_verbosity(tf.logging.INFO)

  # create the estimator
  estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=args.model_dir,
    config=tf.estimator.RunConfig(
      train_distribute=tf.distribute.MirroredStrategy()
        if args.multi_gpu else None),
  )
  # dispatch training
  if args.mode == 'train':
    estimator.train(input_fn, steps=1000000)

  # dispatch generation
  elif args.mode == 'predict':
    output = next(estimator.predict(input_fn))

    # postprocess result
    waveform = dequantize(output, args.quantization)

    # save to .wav file
    save_output(args.output_file, waveform)

def parse_arguments():
  '''Parses command line arguments to wavenet.py
  '''
  p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  sp = p.add_subparsers(dest='mode')
  train = sp.add_parser(
    'train',
    help='train a WaveNet model on some audio data',
  )
  train.add_argument(
    'data_dir',
    type=str,
    help='directory containing .mp3 files to use as training data',
  )
  train.add_argument(
    'model_dir',
    type=str,
    help='directory in which to save checkpoints and summaries',
  )
  train.add_argument(
    '-ch',
    dest='channel_multiplier',
    type=int,
    default=32,
    help='multiplicative factor for all hidden units',
  )
  train.add_argument(
    '-bk',
    dest='blocks',
    type=int,
    default=5,
    help='number of causal blocks in the network',
  )
  train.add_argument(
    '-lp',
    dest='layers_per_block',
    type=int,
    default=13,
    help='number of dilated convolutions per causal block',
  )
  train.add_argument(
    '-bs',
    dest='batch_size',
    type=int,
    default=1,
    help='number of training examples per replica per parameter update',
  )
  train.add_argument(
    '-ls',
    dest='length_secs',
    type=float,
    default=2.,
    help='length in seconds of a single training example',
  )
  train.add_argument(
    '-qz',
    dest='quantization',
    type=int,
    default=256,
    help='number of bins in which to quantize the audio signal',
  )
  train.add_argument(
    '-dy',
    dest='decay',
    type=int,
    default=100000,
    help='number of updates after which to halve the learning rate',
  )
  train.add_argument(
    '-mg',
    dest='multi_gpu',
    action='store_true',
    help='whether to use a MirroredStrategy across all GPUs for training',
  )
  predict = sp.add_parser(
    'predict',
    help='generate audio with a pre-trained WaveNet model'
  )
  predict.add_argument(
    'model_dir',
    type=str,
    help='directory from which to load the latest checkpoint',
  )
  predict.add_argument(
    'output_file',
    type=str,
    help='.wav or .mp3 filepath in which to save the generated waveform',
  )
  predict.add_argument(
    '-ls',
    dest='length_secs',
    type=float,
    default=1.,
    help='length in seconds of the generated waveform',
  )
  return p.parse_args()

if __name__ == '__main__':
  main(parse_arguments())
