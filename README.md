# Installation
1. [Download and build Tensorflow from source](https://www.tensorflow.org/install/source) in order to use the Tensorflow Benchmark Tool. Summarized steps:
    - [Install Bazel](https://docs.bazel.build/versions/master/install.html)
    - Clone the Tensorflow Repository
        - `git clone https://github.com/tensorflow/tensorflow.git`
    - Configure the build
        - `cd tensorflow`
        - `./configure`
    - Build TF 1.X
        - `bazel build --config=v1 //tensorflow/tools/pip_package:build_pip_package`
    - You should the be able to use the [Tensorflow benchmark tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/benchmark) like so:
        - `./tensorflow/bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=decoder.pbtxt --input_layer="input_2:0,input_3:0,input_4:0" --input_layer_shape="1,1,93:1,256:1,256" --input_layer_type=float,float,float --output_layer="dense1_1/truediv:0"`
2. Install this module.
    - Clone the repo.
        - `git clone https://github.com/brian-yu/cnn-rnn-profiling.git`
    - Install the package.
        - `pip install -e ./nn_regressors`

# Using the library

## Instantiate regressor objects
```python
from nn_regressors import CNNRegressor, RNNRegressor
cnn_cpu_reg = CNNRegressor.cpu()
cnn_mem_reg = CNNRegressor.memory()
rnn_cpu_reg = RNNRegressor.cpu()
rnn_mem_reg = RNNRegressor.memory()
```

## Predict layer CPU and memory usage
```python
resnet = tf.keras.applications.resnet50.ResNet50(
    include_top=True,
    weights='imagenet')

layer_cpu_time = cnn_cpu_reg(ResNet50)
```

# Training regression model on your own machine

## Generate Benchmarks

Make sure that your directory structure looks like this:
```
parent_dir/
    tensorflow/
    cnn-rnn-profiling/
```
- ` cd cnn-rnn-profiling`
- `python3 lstm.py`
- `python3 seq2seq.py`
- `python3 inception.py`
- `python3 vgg.py`


<!-- ## Creating Benchmark generation programs -->

## Train the regression model
- `python3 regress.py`

# Limitations
- Currently only works on Keras models.

# Default benchmark machine specs
- 2017 MacBook Pro
- 2.9 GHz Intel Core i7
- 16 GB 2133 MHz DDR3 RAM
- Radeon Pro 560 4 GB GPU (however, Tensorflow CPU was used)