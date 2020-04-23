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
    encoder = LSTM(latent_dim, name="lstm1", return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # enc = Sequential(name='encoder')
    # # model.name = 'lstm'
    # enc.add(Input(shape=(None, num_encoder_tokens)))
    # enc.add(LSTM(latent_dim, name='lstm1', return_sequences=True))
    # model.add(LSTM(32, name='lstm2', return_sequences=True))
    # model.add(LSTM(64, name='lstm3', return_sequences=True))
    # model.add(LSTM(128, name='lstm4', return_sequences=True))
    # model.add(LSTM(48, name='lstm5'))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(
        latent_dim, name="lstm2", return_sequences=True, return_state=True
    )
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, name="dense1", activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    # # Define the model that will turn
    # # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    # model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='seq2seq')

    # # Run training
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
    #             metrics=['accuracy'])
    # return model

    # enc = Model(encoder_inputs, encoder_outputs, name='encoder')
    # enc.compile(optimizer='rmsprop', loss='categorical_crossentropy',
    #     metrics=['accuracy'])
    # dec = Model(decoder_inputs, decoder_outputs, name='decoder')
    # dec.compile(optimizer='rmsprop', loss='categorical_crossentropy',
    #     metrics=['accuracy'])
    # return enc, dec

    encoder_model = Model(encoder_inputs, encoder_states, name="encoder")

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states,
        name="decoder",
    )

    return encoder_model, decoder_model


# model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
#           batch_size=batch_size,
#           epochs=epochs,
#           validation_split=0.2)
# # Save model
# model.save('s2s.h5')

if __name__ == "__main__":

    # model = create_seq2seq()

    # # # tf.keras.backend.clear_session()
    # # sess = tf.keras.backend.get_session()

    # # output_graph_def = convert_variables_to_constants(
    # #     sess,
    # #     sess.graph.as_graph_def(),
    # #     [node.op.name for node in model.outputs])

    # # tf.io.write_graph(output_graph_def, './', f'{model.name}.pbtxt')

    # print(model.name)
    # print(model.inputs)
    # print(model.outputs)

    # # print(model.input[0].shape)
    # # print(model.input[1].shape)
    # print(model.output.name)

    # print(model.summary())

    # print(model.layers)

    enc, dec = create_seq2seq()

    print(enc.summary())
    print(dec.summary())

    print(enc.output)
    print(dec.output)

    print(enc.inputs)
    print(dec.inputs)

    # # tf.keras.backend.clear_session()
    # sess = tf.keras.backend.get_session()

    # output_graph_def = convert_variables_to_constants(
    #     sess,
    #     sess.graph.as_graph_def(),
    #     [node.op.name for node in enc.outputs])

    # tf.io.write_graph(output_graph_def, './', f'{enc.name}.pbtxt')

    # sess = tf.keras.backend.get_session()

    # output_graph_def = convert_variables_to_constants(
    #     sess,
    #     sess.graph.as_graph_def(),
    #     [node.op.name for node in dec.outputs])

    # tf.io.write_graph(output_graph_def, './', f'{dec.name}.pbtxt')


"""

../tensorflow/bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=encoder.pbtxt --input_layer="input_1:0" --input_layer_shape="1,1,71" --output_layer="lstm1/while/Exit_2:0"

../tensorflow/bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=decoder.pbtxt --input_layer="input_2:0,input_3:0,input_4:0" --input_layer_shape="1,1,93:1,256:1,256" --input_layer_type=float,float,float --output_layer="dense1_1/truediv:0"
"""
