"""Based on NELSONZHAO's code(https://github.com/NELSONZHAO/zhihu), perform
   pronunciation prediction using RNN Encoder-Decoder model.
"""

from tensorflow.python.layers.core import Dense
import numpy as np
import tensorflow as tf
import pickle

# Learning rate
learning_rate = 0.001
# Optimizer used by the model, 0 for SGD, 1 for Adam, 2 for RMSProp
optimizer_type = 1
# Mini-batch size
batch_size = 512
# Cell type, 0 for LSTM, 1 for GRU
Cell_type = 0
# Activation function used by RNN cell, 0 for tanh, 1 for relu, 2 for sigmoid
activation_type = 0
# Number of cells in each layer
rnn_size = 128
# Number of layers
num_layers = 2
# Embedding size for encoding part and decoding part
encoding_embedding_size = 128
decoding_embedding_size = encoding_embedding_size
# Decoder type, 0 for basic, 1 for beam search
Decoder_type = 1
# Beam width for beam search decoder
beam_width = 3
# Number of max epochs for training
epochs = 60
# 1 for training, 0 for test the already trained model, 2 for evaluate performance
isTrain = 1
# Display the result of training for every display_step
display_step = 50
# max number of model to keep
max_model_number = 5

# import the data from data.pickle
with open('./dataset/data.pickle', 'r') as f:
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


# create the compute graph
train_graph = tf.Graph()
with train_graph.as_default():
    # define the global step of the graph
    global_step = tf.train.create_global_step(train_graph)
    input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()

    # define the placeholder and summary of the validation loss and WER
    average_vali_loss = tf.placeholder(dtype=tf.float32)
    WER_over_validation = tf.placeholder(dtype=tf.float32)
    v_c = tf.summary.scalar("validation_cost", average_vali_loss)
    v_wer = tf.summary.scalar("validation_WER", WER_over_validation)

    # get the output of the seq2seq model
    training_decoder_output, predicting_decoder_output, bm_decoder_output = seq2seq_model(input_data,
                                                                                          targets,
                                                                                          target_sequence_length,
                                                                                          max_target_sequence_length,
                                                                                          source_sequence_length,
                                                                                          len(source_letter_to_int),
                                                                                          encoding_embedding_size,
                                                                                          decoding_embedding_size,
                                                                                          rnn_size,
                                                                                          num_layers)
    # get the logits of decoder during training and testing to calculate loss.
    training_logits = tf.identity(training_decoder_output.rnn_output, 'training_logits')
    predicting_logits = tf.identity(predicting_decoder_output.rnn_output, 'predicting_logits')
    # the result of the prediction
    prediction = tf.identity(predicting_decoder_output.sample_id, 'prediction_result')
    bm_prediction = tf.identity(bm_decoder_output.predicted_ids, 'bm_prediction_result')
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    # the score of the beam search prediction
    bm_score = tf.identity(bm_decoder_output.beam_search_decoder_output.scores, 'bm_prediction_scores')

    with tf.name_scope("optimization"):
        # Loss function, compute the cross entropy
        train_cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        optimizer_collection = {0: tf.train.GradientDescentOptimizer(lr),
                                1: tf.train.AdamOptimizer(lr),
                                2: tf.train.RMSPropOptimizer(lr)}
        # Using the optimizer defined by optimizer_type
        optimizer = optimizer_collection[optimizer_type]

        # compute gradient
        gradients = optimizer.compute_gradients(train_cost)
        # apply gradient clipping to prevent gradient explosion
        capped_gradients = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gradients if grad is not None]
        # update the RNN
        train_op = optimizer.apply_gradients(capped_gradients, global_step=global_step)
    # define summary to record training cost.
    training_cost_summary = tf.summary.scalar('training_cost', train_cost)

    with tf.name_scope("validation"):
        # get the max length of the predicting result
        val_seq_len = tf.shape(predicting_logits)[1]
        # process the predicting result so that it has the same shape with targets
        predicting_logits = tf.concat([predicting_logits, tf.fill(
            [batch_size, max_target_sequence_length - val_seq_len, tf.shape(predicting_logits)[2]], 0.0)], axis=1)
        # calculate loss
        validation_cost = tf.contrib.seq2seq.sequence_loss(
            predicting_logits,
            targets,
            masks)


def pad_sentence_batch(sentence_batch, pad_int):
    """Pad the mini batch, so that each input/output has the same length
    Args:
        sentence_batch: mini batch to get padded.
        pad_int: an integer representing the symbol of <PAD>
    Returns:
        Padded mini-batch
    """
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(targets, sources, source_pad_int, target_pad_int):
    """Generator to generating the mini batches for training and testing
    Args:
        targets: targets(true result) used for training the decoder, tensor of shape
          [batch_size, max_target_sequence_length].
        sources: input of the model, tensor of shape [batch_size, max_input_length].
        source_pad_int: an integer representing the symbol of <PAD> for input sequence.
        target_pad_int: an integer representing the symbol of <PAD> for output sequence.
    Yields:
        pad_targets_batch: padded targets mini-batch
        pad_sources_batch: padded inputs mini-batch
        targets_length: tensor of shape (mini_batch_size, ), representing the length for
          each target sequence in the mini-batch.
        source_lengths: tensor of shape (mini_batch_size, ), representing the length for
          each input sequence in the mini-batch.
    """
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


train_source = None
train_target = None
valid_source = None
valid_target = None
test_source = None
test_target = None

if isTrain == 1:
    # training the model, make the input and output's size be multiple of mini batch's size.
    train_remainder = len(data_sets['train'][0]) % batch_size
    valid_remainder = len(data_sets['dev'][0]) % batch_size

    train_source = data_sets['train'][0] + data_sets['train'][0][len(data_sets['train'][0]) - train_remainder - 1:]
    train_target = data_sets['train'][1] + data_sets['train'][1][len(data_sets['train'][0]) - train_remainder - 1:]

    valid_source = data_sets['dev'][0] + data_sets['dev'][0][0:batch_size - valid_remainder]
    valid_target = data_sets['dev'][1] + data_sets['dev'][1][0:batch_size - valid_remainder]

elif isTrain == 2:
    test_remainder = len(data_sets['test']) % batch_size
    test_source = data_sets['test'][0] + data_sets['test'][0][len(data_sets['test'][0]) - test_remainder - 1:]
    test_target = data_sets['test'][1] + data_sets['test'][1][len(data_sets['test'][0]) - test_remainder - 1:]


# calculate the error of the prediction
def cal_error(input_batch, prediction_result):
    """Calculate the number of prediction errors
    Args:
        input_batch: the input mini-batch.
        prediction_result: the prediction result of the model.
    Returns:
        t_error: total number of errors across the mini-batch
    """
    t_error = 0.0
    for char_ids, pron_ids in zip(input_batch, prediction_result):
        t_word = map(lambda x: source_int_to_letter[x], char_ids)
        try:
            word = ''.join(t_word[:t_word.index('<PAD>')])
        except ValueError:
            word = ''.join(t_word)
        t_pron = map(lambda x: target_int_to_letter[x], pron_ids)
        try:
            pron = ' '.join(t_pron[:t_pron.index('<EOS>')])
        except ValueError:
            pron = ' '.join(t_pron)
        if pron not in word_pron[word]:
            t_error += 1
    return t_error


# create session to run the TensorFlow operations
with tf.Session(graph=train_graph) as sess:
    # define summary file writer
    t_s = tf.summary.FileWriter('./graph/training', sess.graph)
    v_s = tf.summary.FileWriter('./graph/validation', sess.graph)
    # run initializer
    sess.run(tf.global_variables_initializer())

    # define saver, keep max_model_number of most recent models
    saver = tf.train.Saver(max_to_keep=max_model_number)

    if isTrain == 1:
        # train the model
        for epoch_i in range(1, epochs + 1):
            for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                    get_batches(train_target, train_source,
                                source_letter_to_int['<PAD>'],
                                target_letter_to_int['<PAD>'])):
                # get global step
                step = tf.train.global_step(sess, global_step)
                t_c, _, loss = sess.run(
                    [training_cost_summary, train_op, train_cost],
                    {input_data: sources_batch,
                     targets: targets_batch,
                     lr: learning_rate,
                     target_sequence_length: targets_lengths,
                     source_sequence_length: sources_lengths})

                if batch_i % display_step == 0:
                    # calculate the word error rate (WER) and validation loss of the model
                    error = 0.0
                    vali_loss = []
                    for _, (
                            valid_targets_batch, valid_sources_batch, valid_targets_lengths,
                            valid_source_lengths) in enumerate(
                        get_batches(valid_target, valid_source,
                                    source_letter_to_int['<PAD>'],
                                    target_letter_to_int['<PAD>'])
                    ):
                        validation_loss, basic_prediction = sess.run(
                            [validation_cost, prediction],
                            {input_data: valid_sources_batch,
                             targets: valid_targets_batch,
                             lr: learning_rate,
                             target_sequence_length: valid_targets_lengths,
                             source_sequence_length: valid_source_lengths})

                        vali_loss.append(validation_loss)
                        error += cal_error(valid_sources_batch, basic_prediction)

                    # calculate the average validation cost and the WER over the validation data set
                    vali_loss = sum(vali_loss) / len(vali_loss)
                    WER = error / len(valid_target)
                    vali_summary, wer_summary = sess.run([v_c, v_wer], {average_vali_loss: vali_loss,
                                                                        WER_over_validation: WER
                                                                        })

                    # write the cost to summery
                    t_s.add_summary(t_c, global_step=step)
                    v_s.add_summary(vali_summary, global_step=step)
                    v_s.add_summary(wer_summary, global_step=step)

                    print(
                        'Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  '
                        '- Validation loss: {:>6.3f}'
                        ' - WER: {:>6.2%} '.format(epoch_i,
                                                   epochs,
                                                   batch_i,
                                                   len(train_source) // batch_size,
                                                   loss,
                                                   vali_loss,
                                                   WER))
            # save the model every epoch
            saver.save(sess, save_path='./model/model.ckpt', global_step=step)
        # save the model when finished
        saver.save(sess, save_path='./model/model.ckpt', global_step=step)
        print('Model Trained and Saved')

    else:
        # load model from folder
        checkpoint = tf.train.latest_checkpoint('./model')
        saver.restore(sess, checkpoint)

        # use the trained model to perform pronunciation prediction
        if isTrain == 0:
            while True:
                test_input = raw_input(">>")
                converted_input = [source_letter_to_int[c] for c in test_input] + [
                    source_letter_to_int['<EOS>']]
                # if the decoder type is 0, use the basic decoder, same as set beam width to 0
                if Decoder_type == 0:
                    beam_width = 1
                result = sess.run(
                    [bm_prediction, bm_score, prediction],
                    {input_data: [converted_input] * batch_size,
                     target_sequence_length: [len(converted_input) * 2] * batch_size,
                     source_sequence_length: [len(converted_input)] * batch_size
                     })
                print "result:"
                for i in xrange(beam_width):
                    tmp = []
                    flag = 0
                    for idx in result[0][0, :, i]:
                        tmp.append(target_int_to_letter[idx])
                        if idx == target_letter_to_int['<EOS>']:
                            print ' '.join(tmp)
                            flag = 1
                            break
                    # prediction length exceeds the max length
                    if not flag:
                        print ' '.join(tmp)

                    # print the score of the result
                    print 'score: {0:.4f}'.format(result[1][0, :, i][-1])
                    print ''
        # evaluate the model's performance
        else:
            error = 0.0
            test_loss = []
            for _, (
                    test_targets_batch, test_sources_batch, test_targets_lengths,
                    test_source_lengths) in enumerate(
                get_batches(test_target, test_source,
                            source_letter_to_int['<PAD>'],
                            target_letter_to_int['<PAD>'])
            ):
                validation_loss, basic_prediction = sess.run(
                    [validation_cost, prediction],
                    {input_data: test_sources_batch,
                     targets: test_targets_batch,
                     lr: learning_rate,
                     target_sequence_length: test_targets_lengths,
                     source_sequence_length: test_source_lengths})

                test_loss.append(validation_loss)
                error += cal_error(test_sources_batch, basic_prediction)

            # calculate the average validation cost and the WER over the validation data set
            test_loss = sum(test_loss) / len(test_loss)
            WER = error / len(test_target)
            print('Test loss: {:>6.3f}'
                  ' - WER: {:>6.2%} '.format(test_loss, WER))
