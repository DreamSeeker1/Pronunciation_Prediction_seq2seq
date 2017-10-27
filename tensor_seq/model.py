"""Based on NELSONZHAO's code(https://github.com/NELSONZHAO/zhihu)
   Define the RNN used in this project.
"""

from tensorflow.python.layers.core import Dense
import numpy as np
import tensorflow as tf
import pickle
from params import *

# import the data from data.pickle
with open('./dataset/data.pickle', 'rb') as f:
    source_int_to_letter, source_letter_to_int, \
    target_int_to_letter, target_letter_to_int, data_sets, \
    word_pron = pickle.load(f)


def get_inputs():
    """Generate the tf.placeholder for the model input.

    Returns:
        inputs: input of the model, tensor of shape [batch_size, max_input_length].
        targets: targets(true result) used for training the decoder, tensor of shape
          [batch_size, max_target_sequence_length].
        learning_rate: learning rate for the mini-batch training.
        target_sequence_length: tensor of shape [mini-batch size, ],the length for
          each target sequence in the mini-batch.
        max_target_sequence_length: the max length of target sequence across the
          mini-batch for training.
        source_sequence_length: tensor of shape [mini-batch size, ],the length for
          each input sequence in the mini-batch.

    """

    inputs = tf.placeholder(tf.int32, [batch_size, None], name='inputs')
    targets = tf.placeholder(tf.int32, [batch_size, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    target_sequence_length = tf.placeholder(tf.int32, (batch_size,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (batch_size,), name='source_sequence_length')
    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length


# construct the RNN cell, using LSTM or GRU
def construct_cell(rnn_size, num_layers):
    """Construct multi-layer RNN

    Args:
        rnn_size: the number of hidden units in a single RNN layer.
        num_layers: the number of total layers of the RNN.

    Returns:
        cell: multi-layer Rnn.
    """

    def get_cell(rnn_size):
        """Generate single RNN layer as specified.

        Args:
            rnn_size: the number of hidden units in a single RNN layer.
        Returns:
            A single layer of RNN
        """

        activation_collection = {0: tf.nn.tanh,
                                 1: tf.nn.relu,
                                 2: tf.nn.sigmoid}
        if Cell_type:
            return tf.contrib.rnn.GRUCell(rnn_size, activation=activation_collection[activation_type])
        else:
            return tf.contrib.rnn.LSTMCell(rnn_size, activation=activation_collection[activation_type])

    cell = tf.contrib.rnn.MultiRNNCell([get_cell(rnn_size) for _ in range(num_layers)])
    return cell


def get_encoder_layer(input_data, rnn_size, num_layers,
                      source_sequence_length, source_vocab_size,
                      encoding_embedding_size):
    """Construct the encoder part.

       Args:
           input_data: input of the model, tensor of shape [batch_size, max_input_length].
           rnn_size: the number of hidden units in a single RNN layer.
           num_layers: total number of layers of the encoder.
           source_sequence_length: tensor of shape [mini-batch size, ],the length for
             each input sequence in the mini-batch.
           source_vocab_size: total number of symbols of input sequence.
           encoding_embedding_size: size of embedding for each symbol in input sequence.
       Returns:
           encoder_output: RNN output tensor.
           encoder_state: The final state of RNN
    """
    # Encoder embedding
    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)
    with tf.variable_scope("encoder"):
        cell = construct_cell(rnn_size, num_layers)
        # Performs fully dynamic unrolling of inputs
        encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input,
                                                          sequence_length=source_sequence_length, dtype=tf.float32)
    return encoder_output, encoder_state


# construct the decoder layer
def decoding_layer(target_letter_to_int, decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_length, max_target_sequence_length, encoder_state, decoder_input):
    """Construct the decoding part of the model.
    See the guide https://www.tensorflow.org/versions/master/api_guides/python/contrib.seq2seq#Dynamic_Decoding

    Args:
        target_letter_to_int: mapping target sequence symbol to int, dict {symbol:int}.
        decoding_embedding_size: target symbol embedding size.
        num_layers: total number of layers of the decoder.
        rnn_size: the number of hidden units in a single RNN layer.
        target_sequence_length: tensor of shape [mini-batch size, ],the length for
          each target sequence in the mini-batch.
        max_target_sequence_length: the max length of target sequence across the
          mini-batch for training.
        encoder_state: the final state of encoder, feeds to decoder as initial state.
        decoder_input: tensor of shape [mini_batch_size, max_target_sequence_length],
          true result for training.
    Returns:
        training_decoder_output: final output of the decoder during training.
        predicting_decoder_output: final output of the decoder during validation.
        bm_decoder_output: final output of the beam search decoder.
    """
    # Embedding the output sequence
    target_vocab_size = len(target_letter_to_int)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

    # construct RNN layer for decoder
    cell = construct_cell(rnn_size, num_layers)

    # output fully connected to last layer, default using linear activation.
    output_layer = Dense(target_vocab_size,
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), name="dense_layer")

    # Training the decoder
    with tf.variable_scope("decoder"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                           training_helper,
                                                           encoder_state,
                                                           output_layer)
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                          impute_finished=True,
                                                                          maximum_iterations=max_target_sequence_length)

    # testing the model, reuse the variables of the trained model
    with tf.variable_scope("decoder", reuse=True):
        start_tokens = tf.tile([tf.constant(target_letter_to_int['<GO>'], dtype=tf.int32)], [batch_size],
                               name='start_tokens')
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                     start_tokens,
                                                                     target_letter_to_int['<EOS>'])

        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                             predicting_helper,
                                                             encoder_state,
                                                             output_layer)
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                            maximum_iterations=max_target_sequence_length)

        tiled_encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, beam_width)
        bm_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell, decoder_embeddings, start_tokens,
                                                          target_letter_to_int['<EOS>'], tiled_encoder_state,
                                                          beam_width, output_layer)

        # impute_finished must be set to false when using beam search decoder
        # https://github.com/tensorflow/tensorflow/issues/11598
        bm_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(bm_decoder,
                                                                    maximum_iterations=max_target_sequence_length)
    return training_decoder_output, predicting_decoder_output, bm_decoder_output


def process_decoder_input(targets, vocab_to_int):
    """Process the target sequence as input to train the model.
    a. cut the last symbol of the target since it won't be fed to the network (<EOS>, <PAD>).
    b. add <GO> to each sequence.

    Args:
        targets: targets(true result) used for training the decoder, tensor of shape
          [batch_size, max_target_sequence_length].
        vocab_to_int: dict {output_symbol:int}, mapping output symbol to int.
    Returns:
        decoder_input: the already processed target sequence
    """
    ending = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return decoder_input


def seq2seq_model(input_data, targets, target_sequence_length,
                  max_target_sequence_length, source_sequence_length,
                  source_vocab_size,
                  encoder_embedding_size, decoding_embedding_size,
                  rnn_size, num_layers):
    """Construct the seq2seq model by connecting encoder part and decoder part.

    Args:
        input_data: input of the model, tensor of shape [batch_size, max_input_length].
        targets: targets(true result) used for training the decoder, tensor of shape
          [batch_size, max_target_sequence_length].
        target_sequence_length:
        max_target_sequence_length:
        source_sequence_length: tensor of shape [mini-batch size, ],the length for
          each input sequence in the mini-batch.
        source_vocab_size: total number of symbols of input sequence.
        encoder_embedding_size: size of embedding for each symbol in input sequence.
        decoding_embedding_size: size of embedding for each symbol in target sequence.
        rnn_size: the number of hidden units in a single RNN layer.
        num_layers: total number of layers of the encoder.
    Returns:
        training_decoder_output: final output of the decoder during training.
        predicting_decoder_output: final output of the decoder during validation.
        bm_decoder_output: final output of the beam search decoder.
    """
    _, encoder_state = get_encoder_layer(input_data,
                                         rnn_size,
                                         num_layers,
                                         source_sequence_length,
                                         source_vocab_size,
                                         encoder_embedding_size)

    decoder_input = process_decoder_input(targets, target_letter_to_int)

    training_decoder_output, predicting_decoder_output, bm_decoder_output = decoding_layer(target_letter_to_int,
                                                                                           decoding_embedding_size,
                                                                                           num_layers,
                                                                                           rnn_size,
                                                                                           target_sequence_length,
                                                                                           max_target_sequence_length,
                                                                                           encoder_state,
                                                                                           decoder_input)

    return training_decoder_output, predicting_decoder_output, bm_decoder_output
