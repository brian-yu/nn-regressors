from tensorflow.keras import layers
import tensorflow.compat.v1 as tf
import subprocess


# This line is required in order for tf.io.write_graph() to work.
# This problem occurs in all keras models that use batch norm.
# See: https://github.com/tensorflow/tensorflow/issues/31331
tf.keras.backend.set_learning_phase(0)

model = tf. keras.applications.inception_v3.InceptionV3(
    include_top=True, weights='imagenet')


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
        '../tensorflow/bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=inception.pbtxt --input_layer="input_1:0" --input_layer_shape="1,299,299,3" --output_layer="predictions/Softmax:0"'
    ], stderr=subprocess.PIPE, shell=True)


output = benchmark.stderr.decode('unicode_escape')

output = output[output.find('Run Order'):output.find('Top by Computation Time')].split('\n')[2:-2]

print(output)


# with tf.keras.backend.get_session() as sess:

#     output_graph_def = tf.graph_util.convert_variables_to_constants(
#         sess,
#         sess.graph.as_graph_def(),
#         [node.op.name for node in model.outputs])

#     tf.io.write_graph(output_graph_def, './', 'inception.pbtxt')

"""
How to run with benchmark tool:

bazel-bin/tensorflow/tools/benchmark/benchmark_model \
    --graph=../models/inception.pbtxt \
    --input_layer="input_1:0" \
    --input_layer_shape="1,299,299,3" \
    --output_layer="predictions/Softmax:0" 

where input_layer and output_layer are given by model.input and model.output

"""