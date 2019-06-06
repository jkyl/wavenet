# WaveNet
[_cf. the original paper here_](https://arxiv.org/abs/1609.03499)

## Setup
```
$ python3 -m pip install -r requirements.txt
```

## Usage
### Synposis
```
$ python3 wavenet.py --help
usage: wavenet.py [-h] {train,predict} ...

positional arguments:
  {train,predict}
    train          train a WaveNet model on some audio data
    predict        generate audio with a pre-trained WaveNet model

optional arguments:
  -h, --help       show this help message and exit
```
### Training
```
$ python3 wavenet.py train --help
usage: wavenet.py train [-h] [-ch CHANNEL_MULTIPLIER] [-bk BLOCKS]
                        [-lp LAYERS_PER_BLOCK] [-bs BATCH_SIZE]
                        [-ls LENGTH_SECS] [-qz QUANTIZATION] [-dy DECAY] [-mg]
                        data_dir model_dir

positional arguments:
  data_dir              directory containing .mp3 files to use as training
                        data
  model_dir             directory in which to save checkpoints and summaries

optional arguments:
  -h, --help            show this help message and exit
  -ch CHANNEL_MULTIPLIER
                        multiplicative factor for all hidden units
  -bk BLOCKS            number of causal blocks in the network
  -lp LAYERS_PER_BLOCK  number of dilated convolutions per causal block
  -bs BATCH_SIZE        number of training examples per replica per parameter
                        update
  -ls LENGTH_SECS       length in seconds of a single training example
  -qz QUANTIZATION      number of bins in which to quantize the audio signal
  -dy DECAY             number of updates after which to halve the learning
                        rate
  -mg                   whether to use a MirroredStrategy across all GPUs for
                        training
```
### Prediction
```
$ python3 wavenet.py predict --help
usage: wavenet.py predict [-h] [-ls LENGTH_SECS] model_dir output_file

positional arguments:
  model_dir        directory from which to load the latest checkpoint
  output_file      .wav or .mp3 filepath in which to save the generated
                   waveform

optional arguments:
  -h, --help       show this help message and exit
  -ls LENGTH_SECS  length in seconds of the generated waveform
```
