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
        - `git clone https://github.com/brian-yu/nn-regressors.git`
    - Install the package.
        - `pip install -e ./`

After these steps, your directory structure looks like this:
```
parent_dir/
    tensorflow/
    nn-regressors/
```

# Using the library

## Instantiate regressor objects
```python
from nn_regressors import CNNRegressor, RNNRegressor

cnn_cpu_reg = CNNRegressor.CPU()
cnn_mem_reg = CNNRegressor.Memory()

rnn_cpu_reg = RNNRegressor.CPU()
rnn_mem_reg = RNNRegressor.Memory()
```

## Predict layer CPU and memory usage
```python
# Instantiate example model.
resnet = tf.keras.applications.resnet50.ResNet50(
    include_top=True,
    weights='imagenet')

# Both method calls return a Numpy array containing predictions for each layer.
layer_cpu_time = cnn_cpu_reg(resnet)
layer_mem_usage = cnn_mem_reg(resnet)
```

# Training regression model on your own machine

## Generate Benchmarks

Make sure that your directory structure looks like this:
```
parent_dir/
    tensorflow/
    nn-regressors/
```

```bash
$ cd nn-regressors

$ python3 lstm.py
$ python3 seq2seq.py
$ python3 inception.py
$ python3 vgg.py
```


<!-- ## Creating Benchmark generation programs -->

## Train the regression model
```bash
$ python3 regress.py
```

# Limitations
- Currently only works on Keras models.
- Complex model architectures may be problematic when used with Tensorflow Benchmark Tool

# Default benchmark machine specs
- 2017 MacBook Pro
- 2.9 GHz Intel Core i7
- 16 GB 2133 MHz DDR3 RAM
- Radeon Pro 560 4 GB GPU (however, Tensorflow CPU was used)