# LSTM and CNN for sequence classification in the IMDB dataset
import numpy

import tensorflow.compat.v1 as tf

from tensorflow.compat.v1.keras.datasets import imdb
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense
from tensorflow.compat.v1.keras.layers import LSTM
from tensorflow.compat.v1.keras.layers import Conv1D
from tensorflow.compat.v1.keras.layers import MaxPooling1D
from tensorflow.compat.v1.keras.layers import Embedding
from tensorflow.compat.v1.keras.preprocessing import sequence

from tf_graph_util import convert_variables_to_constants


# fix random seed for reproducibility
numpy.random.seed(7)

# constants
top_words = 5000
max_review_length = 500

def create_lstm():
    # create the model
    embedding_vecor_length = 32
    model = Sequential(name='lstm')
    # model.name = 'lstm'
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(10, name='lstm1', return_sequences=True))
    model.add(LSTM(32, name='lstm2', return_sequences=True))
    model.add(LSTM(64, name='lstm3', return_sequences=True))
    model.add(LSTM(128, name='lstm4', return_sequences=True))
    model.add(LSTM(48, name='lstm5'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model




if __name__ == '__main__':

    model = create_lstm()

    # # load the dataset but only keep the top n words, zero the rest
    # (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

    # # truncate and pad input sequences
    # X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    # X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    # print(model.summary())
    # model.fit(X_train, y_train, epochs=3, batch_size=64)
    # # Final evaluation of the model
    # scores = model.evaluate(X_test, y_test, verbose=0)
    # print("Accuracy: %.2f%%" % (scores[1]*100))
 

    '''
        # tf.keras.backend.clear_session()
        sess = tf.keras.backend.get_session()

        # print(sess.graph.as_graph_def())
        # print(tf.keras.backend.set_learning_phase(0))

        output_graph_def = convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(),
            [node.op.name for node in model.outputs])

        tf.io.write_graph(output_graph_def, './', f'{model.name}.pbtxt')
    '''

    print(model.summary())

    print(model.inputs)
    print(model.layers)
    print(model.outputs)