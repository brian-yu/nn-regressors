# LSTM and CNN for sequence classification in the IMDB dataset

import tensorflow.compat.v1 as tf


from tensorflow.compat.v1.keras.models import Sequential, Model
from tensorflow.compat.v1.keras.layers import Dense, LSTM, Input, Concatenate

from tf_graph_util import convert_variables_to_constants

# from keras.layers import Input, LSTM, Dense
import numpy as np

def create_seq2seq():
    batch_size = 64  # Batch size for training.
    epochs = 100  # Number of epochs to train for.
    latent_dim = 256  # Latent dimensionality of the encoding space.
    num_samples = 10000  # Number of samples to train on.

    num_encoder_tokens = 71
    num_decoder_tokens = 93

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, name='lstm1', return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, name='lstm2', return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                        initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, name='dense1', activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='seq2seq')

    # Run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model



# model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
#           batch_size=batch_size,
#           epochs=epochs,
#           validation_split=0.2)
# # Save model
# model.save('s2s.h5')

if __name__ == '__main__':

    model = create_seq2seq()


    # # tf.keras.backend.clear_session()
    # sess = tf.keras.backend.get_session()

    # output_graph_def = convert_variables_to_constants(
    #     sess,
    #     sess.graph.as_graph_def(),
    #     [node.op.name for node in model.outputs])

    # tf.io.write_graph(output_graph_def, './', f'{model.name}.pbtxt')

    print(model.name)
    print(model.inputs)
    print(model.outputs)


    # print(model.input[0].shape)
    # print(model.input[1].shape)
    print(model.output.name)

    print(model.summary())

    print(model.layers)




"""
../tensorflow/bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph={model.name}.pbtxt --input_layer="{model.input.name}" --input_layer_shape="{input_shape}" --output_layer="{model.output.name}"


../tensorflow/bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=seq2seq.pbtxt --input_layer="input_1:0" --input_layer_shape="1,1,71" --output_layer="dense1/truediv:0"

../tensorflow/bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=seq2seq.pbtxt --input_layer="input_2:0" --input_layer_shape="1,1,93" --output_layer="dense1/truediv:0"
"""