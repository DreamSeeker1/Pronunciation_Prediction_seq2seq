from tensorflow.python.layers.core import Dense
import numpy as np
import time
import tensorflow as tf

# open the file contains data
with open('/home/yy/pronunciation-prediction/tensor_seq/source_list') as f:
    source_data = f.read().split('\n')
with open('/home/yy/pronunciation-prediction/tensor_seq/target_list') as f:
    target_data = f.read().split('\n')

# parameters
# Number of Epochs
epochs = 2
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 128
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 20
decoding_embedding_size = 20
# Learning Rate
learning_rate = 0.001
# cell type 0 for lstm, 1 for GRU
C_type = 0
# decoder type 0 for basic, 1 for beam search
D_type = 0


# map the character and pronunciation to idx
def word2index(data):
    special_words = ['<EOS>', '<GO>', '<PAD>', '<UNK>']
    set_words = list(set([c for line in data for c in line] + special_words))

    idx2word = {idx: word for idx, word in enumerate(set_words)}
    word2idx = {word: idx for idx, word in idx2word.items()}
    return idx2word, word2idx


def phon2index(data):
    special_words = ['<EOS>', '<GO>', '<PAD>', '<UNK>']
    set_words = list(
        set([pho for line in data for pho in line.split()] + special_words))

    idx2word = {idx: word for idx, word in enumerate(set_words)}
    word2idx = {word: idx for idx, word in idx2word.items()}
    return idx2word, word2idx


source_int_to_letter, source_letter_to_int = word2index(source_data)
target_int_to_letter, target_letter_to_int = phon2index(target_data)

source_int = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>'])
               for letter in line] for line in source_data]
target_int = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>'])
               for letter in line.split()] + [target_letter_to_int['<EOS>']] for line in target_data]


# input of the model
def get_inputs():
    # input
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    # targets used for training the decoder
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    # target seq length
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    # max target seq length
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    # max source sequence length
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length


# construct the RNN cell, using LSTM or GRU
def construct_cell(rnn_size, num_layers, cell_type):
    def get_cell(rnn_size, cell_type):
        if cell_type == 0:
            cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        else:
            cell = tf.contrib.rnn.GRUCell(rnn_size)
        return cell

    cell = tf.contrib.rnn.MultiRNNCell([get_cell(rnn_size, cell_type) for _ in range(num_layers)])
    return cell


# construct encoder layer
def get_encoder_layer(input_data, rnn_size, num_layers,
                      source_sequence_length, source_vocab_size,
                      encoding_embedding_size, cell_type):
    # Encoder embedding
    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)

    # RNN cell
    cell = construct_cell(rnn_size, num_layers, cell_type)

    encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input,
                                                      sequence_length=source_sequence_length, dtype=tf.float32)

    return encoder_output, encoder_state


def decoding_layer(target_letter_to_int, decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_length, max_target_sequence_length, encoder_state, decoder_input, cell_type,
                   decoder_type):
    # Embedding
    target_vocab_size = len(target_letter_to_int)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

    # construct cell
    cell = construct_cell(rnn_size, num_layers, cell_type)

    # output fully connected layer
    output_layer = Dense(target_vocab_size,
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    # Training decoder
    with tf.variable_scope("decode"):
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
    # Predicting decoder
    # sharing parameters
    with tf.variable_scope("decode", reuse=True):
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
                                                                            impute_finished=True,
                                                                            maximum_iterations=max_target_sequence_length)

    return training_decoder_output, predicting_decoder_output


def process_decoder_input(data, vocab_to_int, batch_size):
    ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return decoder_input


def seq2seq_model(input_data, targets, lr, target_sequence_length,
                  max_target_sequence_length, source_sequence_length,
                  source_vocab_size, target_vocab_size,
                  encoder_embedding_size, decoder_embedding_size,
                  rnn_size, num_layers, cell_type, decoder_type):
    _, encoder_state = get_encoder_layer(input_data,
                                         rnn_size,
                                         num_layers,
                                         source_sequence_length,
                                         source_vocab_size,
                                         encoder_embedding_size, cell_type)

    decoder_input = process_decoder_input(targets, target_letter_to_int, batch_size)

    training_decoder_output, predicting_decoder_output = decoding_layer(target_letter_to_int,
                                                                        decoding_embedding_size,
                                                                        num_layers,
                                                                        rnn_size,
                                                                        target_sequence_length,
                                                                        max_target_sequence_length,
                                                                        encoder_state,
                                                                        decoder_input, cell_type,
                                                                        decoder_type)

    return training_decoder_output, predicting_decoder_output


train_graph = tf.Graph()

with train_graph.as_default():
    input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()
    cell_type = tf.placeholder(tf.int32, name='cell_type')
    decoder_type = tf.placeholder(tf.int32, name='decoder_type')
    training_decoder_output, predicting_decoder_output = seq2seq_model(input_data,
                                                                       targets,
                                                                       lr,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       source_sequence_length,
                                                                       len(source_letter_to_int),
                                                                       len(target_letter_to_int),
                                                                       encoding_embedding_size,
                                                                       decoding_embedding_size,
                                                                       rnn_size,
                                                                       num_layers, cell_type, decoder_type)

    training_logits = tf.identity(training_decoder_output.rnn_output, 'training')
    predicting_logits = tf.identity(predicting_decoder_output.rnn_output, 'predictions')

    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        train_cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(train_cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
        tf.summary.scalar('training_cost', train_cost)

    with tf.name_scope("validation"):
        # get sequence length
        val_seq_len = tf.shape(predicting_logits)[1]
        predicting_logits = tf.concat([predicting_logits, tf.fill(
            [batch_size, max_target_sequence_length - val_seq_len, tf.shape(predicting_logits)[2]], 0.0)], axis=1)
        # Loss function
        validation_cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)
        tf.summary.scalar('validation_cost', validation_cost)


def pad_sentence_batch(sentence_batch, pad_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

        yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths


train_source = source_int[batch_size:]
train_target = target_int[batch_size:]

valid_source = source_int[:batch_size]
valid_target = target_int[:batch_size]
(valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(
    get_batches(valid_target, valid_source, batch_size,
                source_letter_to_int['<PAD>'],
                target_letter_to_int['<PAD>']))

display_step = 50

checkpoint = "trained_model.ckpt"

with tf.Session(graph=train_graph) as sess:
    # define summary
    merged = tf.summary.merge_all()
    Summ = tf.summary.FileWriter('./graph', graph=sess.graph)
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(1, epochs + 1):
        for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                get_batches(train_target, train_source, batch_size,
                            source_letter_to_int['<PAD>'],
                            target_letter_to_int['<PAD>'])):
            _, loss = sess.run(
                [train_op, train_cost],
                {input_data: sources_batch,
                 targets: targets_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 source_sequence_length: sources_lengths,
                 cell_type: C_type,
                 decoder_type: D_type})

            if batch_i % display_step == 0:
                summary, validation_loss = sess.run(
                    [merged, validation_cost],
                    {input_data: valid_sources_batch,
                     targets: valid_targets_batch,
                     lr: learning_rate,
                     target_sequence_length: valid_targets_lengths,
                     source_sequence_length: valid_sources_lengths,
                     cell_type: C_type,
                     decoder_type: D_type
                     })

                # log the data
                Summ.add_summary(summary)

                print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                      .format(epoch_i,
                              epochs,
                              batch_i,
                              len(train_source) // batch_size,
                              loss,
                              validation_loss))

    saver = tf.train.Saver()
    saver.save(sess, checkpoint)
    print('Model Trained and Saved')
