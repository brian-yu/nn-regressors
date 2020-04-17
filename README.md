## Table of Contents
1. [Installation](#Installation)
2. [Using the Library](#Using-the-library)
    - [Instantiate regressor objects](#Instantiate-regressor-objects)
    - [Predict layer CPU and memory usage](#Predict-layer-CPU-and-memory-usage)
    - [Training regression model on your own machine](#Training-regression-model-on-your-own-machine)
3. [Limitations](#Limitations)
4. [Default Benchmark Machine Specs](#Default-benchmark-machine-specs)
5. [Running Examples](#Running-Examples)
6. [Important Files](#Important-Files)

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
        - `pip install -e nn-regressors`

After these steps, your directory structure should look like this:

```bash
parent_dir/
    tensorflow/
    nn-regressors/
```

Create a new directory on the same level as `tensorflow/` in which you will do your work. This is important because the benchmark scripts use `subprocess` to run benchmarks (see [example](https://github.com/brian-yu/nn-regressors/blob/d0e1063a1b3894d27db376e00c8d367c5b2ae555/nn_regressors/benchmark.py#L30)).

```bash
parent_dir/
    tensorflow/
    nn-regressors/
    work_dir/ # create this!
```

# Using the library

## Instantiate regressor objects
```python
from nn_regressors import CNN, RNN

cnn_cpu_reg = CNN.CPURegressor()
cnn_mem_reg = CNN.MemoryRegressor()

rnn_cpu_reg = RNN.CPURegressor()
rnn_mem_reg = RNN.MemoryRegressor()
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

### Generate Benchmarks (simple models)

Make sure that your directory structure looks like this and that you have built `tensorflow` from source using `bazel`:
```
parent_dir/
    tensorflow/
    nn-regressors/
    work_dir_arbitrary_name/ # put code that uses the library here!
```

You can add new models programatically:

```python
import tensorflow.compat.v1 as tf
from nn_regressors import CNN

# Create Regressor instances
cnn_cpu_reg = CNN.CPURegressor()
cnn_mem_reg = CNN.MemoryRegressor()

# Load VGG model.
tf.keras.backend.clear_session() # IMPORTANT for layer names to match up.
vgg16_model = tf.keras.applications.vgg16.VGG16(
    include_top=True,
    weights='imagenet',
)

# Add vgg model data
cnn_cpu_reg.add_model_data(vgg16_model)
cnn_mem_reg.add_model_data(vgg16_model)

# Load inception model.
tf.keras.backend.clear_session() # IMPORTANT for layer names to match up.
inception_model = tf. keras.applications.inception_v3.InceptionV3(
    include_top=True,
    weights='imagenet',
)

# Add inception model data
cnn_cpu_reg.add_model_data(inception_model)
cnn_mem_reg.add_model_data(inception_model)

# Fit regressors to new model data.
cnn_cpu_reg.fit()
cnn_mem_reg.fit()


## Evaluate on new model.
mobilenet = tf.keras.applications.mobilenet_v2.MobileNetV2(
    include_top=True,
    weights='imagenet')

# Print MSEs of regression models.
print(cnn_cpu_reg.evaluate(mobilenet))
print(cnn_mem_reg.evaluate(mobilenet))

# You can also save the model state.
# After calling save() with no arguments, future instances
# will automatically load the latest saved model.
cnn_cpu_reg.save()
cnn_mem_reg.save()

load_prev_save_cpu_reg = CNN.CPURegressor()

# If you call save(filename) with an arg, it will save to that specific
# file, which you can load later.
cnn_cpu_reg.save('reg1.joblib')
new_cnn_cpu_reg = CNN.CPURegressor(save_file='reg1.joblib')
```

**Note: You can clear saved benchmark results with `rm *_benchmark.txt` and clear serialized models with `rm *.pbtxt`.**

### Generating Benchmarks for Complex Models
For more complex models (e.g. seq2seq) on which the above code fails, you may have to run the benchmarks manually first.

This will create `<model_name>_benchmark.txt` files that the library will use instead of calling the Tensorflow Benchmark Tool.

#### Example:

For the seq2seq model in [seq2seq.py](https://github.com/brian-yu/nn-regressors/blob/d0e1063a1b3894d27db376e00c8d367c5b2ae555/seq2seq.py), the model must be split into 2; the encoder and the decoder model.

1. Serialize the models.

```python
import tensorflow.compat.v1 as tf

from seq2seq import create_seq2seq
from nn_regressors import 

tf.keras.backend.clear_session()
enc, dec = create_seq2seq()

# Serialize encoder into `encoder.pbtxt`
sess = tf.keras.backend.get_session()
output_graph_def = convert_variables_to_constants(
    sess,
    sess.graph.as_graph_def(),
    [node.op.name for node in enc.outputs])
tf.io.write_graph(output_graph_def, './', f'{enc.name}.pbtxt')

# Serialize decoder into `decoder.pbtxt`
sess = tf.keras.backend.get_session()
output_graph_def = convert_variables_to_constants(
    sess,
    sess.graph.as_graph_def(),
    [node.op.name for node in dec.outputs])
tf.io.write_graph(output_graph_def, './', f'{dec.name}.pbtxt')

# Print the model summaries in order to find out the input layer name,
# the output layer name, and their shapes. This will be useful
# when running the Tensorflow benchmark tool.
print(enc.summary())
print(dec.summary())

```

2. Run the benchmarks.

In order to figure out the arguments for the Tensorflow benchmark tool, look at the Keras model summaries.

```bash
#  Benchmark the encoder.
$ ../tensorflow/bazel-bin/tensorflow/tools/benchmark/benchmark_model \
    --graph=encoder.pbtxt \
    --input_layer="input_1:0" \
    --input_layer_shape="1,1,71" \
    --output_layer="lstm1/while/Exit_2:0"

# Benchmark the decoder.
$ ../tensorflow/bazel-bin/tensorflow/tools/benchmark/benchmark_model \
    --graph=decoder.pbtxt \
    --input_layer="input_2:0,input_3:0,input_4:0" \
    --input_layer_shape="1,1,93:1,256:1,256" \
    --input_layer_type=float,float,float \
    --output_layer="dense1_1/truediv:0"
```

3. Add model data as above.
Since the model benchmark data has been saved in `<model_name>_benchmark.txt`, you can run the code in the previous section, using `regressor.add_model_data(model)`.

# Limitations
- Currently only works on Keras models.
- Complex model architectures may be problematic when used with Tensorflow Benchmark Tool

# Default benchmark machine specs
- 2017 MacBook Pro
- 2.9 GHz Intel Core i7
- 16 GB 2133 MHz DDR3 RAM
- Radeon Pro 560 4 GB GPU (however, Tensorflow CPU was used)

# Running Examples
Run examples in the `examples/` directory from the main folder. e.g.
```bash
$ python examples/example.py
```

# Important Files
- `nn_regressors/regressors.py`: Regressor fitting and evaluation code. You can tweak the regression model by editing this file.
- `nn_regressors/benchmark.py`: Code that runs the Tensorflow Benchmark Tool.
- `nn_regressors/utils.py`: Code that cleans and preprocesses data.