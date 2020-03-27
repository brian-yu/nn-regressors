import io
import subprocess

from tensorflow.keras import layers
import tensorflow.compat.v1 as tf
import pandas as pd


model = tf.keras.applications.vgg16.VGG16(
    include_top=True,
    weights='imagenet')

print(model.input)
print(model.output)
print([node.op.name for node in model.outputs])

print(model.summary())

print("========")
print(model.layers[1])
print(model.layers[1].get_config())

print("========")

print(model.layers[-3])
print(model.layers[-3].get_config())

print("========")

print(model.layers[-1])
print(model.layers[-1].get_config())

print("========")

print(model.input.name)
print(model.input.shape)
print(model.output.name)

benchmark = subprocess.run(
    [
        '../tensorflow/bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=vgg.pbtxt --input_layer="input_1:0" --input_layer_shape="1,224,224,3" --output_layer="predictions/Softmax:0"'
    ], stderr=subprocess.PIPE, shell=True)


output = benchmark.stderr.decode('unicode_escape')

split_output = output[output.find('Run Order'):output.find('Top by Computation Time')].split('\n')

f = io.StringIO("\n".join(split_output[1:-2]))
df = pd.DataFrame.read_csv(f, sep="\t")
print(df.head())


print(split_output[2:-2])


# with tf.keras.backend.get_session() as sess:
#     output_graph_def = tf.graph_util.convert_variables_to_constants(
#         sess,
#         sess.graph.as_graph_def(),
#         [node.op.name for node in model.outputs])

#     tf.io.write_graph(output_graph_def, './', 'vgg.pbtxt')

"""
How to run with benchmark tool:

bazel-bin/tensorflow/tools/benchmark/benchmark_model \
    --graph=../models/vgg.pbtxt \
    --input_layer="input_1:0" \
    --input_layer_shape="1,224,224,3" \
    --output_layer="predictions/Softmax:0" 

where input_layer and output_layer are given by model.input and model.output

"""