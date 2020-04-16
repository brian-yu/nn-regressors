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
from nn_regressors import cnn, rnn

cnn_cpu_reg = cnn.CPURegressor()
cnn_mem_reg = cnn.MemoryRegressor()

rnn_cpu_reg = rnn.CPURegressor()
rnn_mem_reg = rnn.MemoryRegressor()
```

## Predict layer CPU and memory usage
```python
import tensorflow.compat.v1 as tf

# Instantiate example model.
resnet = tf.keras.applications.resnet50.ResNet50(
    include_top=True,
    weights='imagenet')

# Both method calls return a Pandas dataframe containing predictions for each layer.
layer_cpu_time = cnn_cpu_reg.predict(resnet)
layer_mem_usage = cnn_mem_reg.predict(resnet)
```

## Training regression model on your own machine

## Generate Benchmarks

Make sure that your directory structure looks like this and that you have built `tensorflow` from source using `bazel`:
```
parent_dir/
    tensorflow/
    nn-regressors/
```

```python
from nn_regressors import cnn, rnn

# Create Regressor instances
cnn_cpu_reg = cnn.CPURegressor()
cnn_mem_reg = cnn.MemoryRegressor()

# Load VGG model.
tf.keras.backend.clear_session() # IMPORTANT for layer names to match up.
vgg16_model = tf.keras.applications.vgg16.VGG16(
    include_top=True,
    weights='imagenet')

# Add vgg model data
cnn_cpu_reg.add_model_data(vgg16_model)
cnn_mem_reg.add_model_data(vgg16_model)

# Load inception model.
tf.keras.backend.clear_session() # IMPORTANT for layer names to match up.
inception_model = tf. keras.applications.inception_v3.InceptionV3(
    include_top=True, weights='imagenet')

# Add inception model data
cnn_cpu_reg.add_model_data(inception_model)
cnn_mem_reg.add_model_data(inception_model)

# Fit regressors to new model data.
cnn_cpu_reg.fit()
cnn_mem_reg.fit()

# You can also save the model state.
# After calling save() with no arguments, future instances
# will automatically load the latest saved model.
cnn_cpu_reg.save()
cnn_mem_reg.save()

load_prev_save_cpu_reg = cnn.CPURegressor()

# If you call save(filename) with an arg, it will save to that specific
# file, which you can load later.
cnn_cpu_reg.save('reg1.joblib')
new_cnn_cpu_reg = cnn.CPURegressor(save_file='reg1.joblib')
```

#### NOTE: For more complex models (e.g. seq2seq), you may have to run the benchmarks individually first.

# Limitations
- Currently only works on Keras models.
- Complex model architectures may be problematic when used with Tensorflow Benchmark Tool

# Default benchmark machine specs
- 2017 MacBook Pro
- 2.9 GHz Intel Core i7
- 16 GB 2133 MHz DDR3 RAM
- Radeon Pro 560 4 GB GPU (however, Tensorflow CPU was used)