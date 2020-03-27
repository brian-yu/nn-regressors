from tensorflow.keras import layers
import tensorflow.compat.v1 as tf


print(tf.version.VERSION)
print(tf.keras.__version__)

model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(1024, activation='relu', input_shape=(32,)),
# Add another:
layers.Dense(1024, activation='relu'),
# Add a softmax layer with 10 output units:
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


import numpy as np

def random_one_hot_labels(shape):
  n, n_class = shape
  classes = np.random.randint(0, n_class, n)
  labels = np.zeros((n, n_class))
  labels[np.arange(n), classes] = 1
  return labels

data = np.random.random((1000, 32))
labels = random_one_hot_labels((1000, 10))

model.fit(data, labels, epochs=10, batch_size=32)

print(model.input)
print(model.output)
print([node.op.name for node in model.outputs])




with tf.keras.backend.get_session() as sess:


  print(sess.graph.get_operations())


  output_graph_def = tf.graph_util.convert_variables_to_constants(
    sess,
    sess.graph.as_graph_def(),
    [node.op.name for node in model.outputs])

  tf.io.write_graph(output_graph_def, './', 'model.pbtxt')


"""
How to run with benchmark tool:

bazel-bin/tensorflow/tools/benchmark/benchmark_model \
    --graph=../models/model.pbtxt \
    --input_layer="dense_input:0" \
    --input_layer_shape="1,32" \
    --output_layer="dense_2/Softmax:0" 

where input_layer and output_layer are given by model.input and model.output

"""