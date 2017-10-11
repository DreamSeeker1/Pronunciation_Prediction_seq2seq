from tensorflow.python.layers.core import Dense
import numpy as np
import tensorflow as tf
import pickle

# parameters

# Number of Epochs
epochs = 60
# Batch Size
batch_size = 256
# RNN Size
rnn_size = 256
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 256
decoding_embedding_size = encoding_embedding_size
# Learning Rate
learning_rate = 0.001
# cell type 0 for lstm, 1 for GRU
Cell_type = 0
# decoder type 0 for basic, 1 for beam search
Decoder_type = 1
# beam width for beam search decoder
beam_width = 3
# 1 for training, 0 for test the already trained model
isTrain = 1
# display step for training
display_step = 50

# import the data
with open('data.pickle', 'r') as f:
    source_int_to_letter, source_letter_to_int, \
    target_int_to_letter, target_letter_to_int, source_int, target_int, \
    word_pron = pickle.load(f)


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
def construct_cell(rnn_size, num_layers):
    def get_cell(rnn_size):
        if Cell_type:
            return tf.contrib.rnn.GRUCell(rnn_size)
        else:
            return tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))

    cell = tf.contrib.rnn.MultiRNNCell([get_cell(rnn_size) for _ in range(num_layers)])
    return cell


# construct encoder layer
def get_encoder_layer(input_data, rnn_size, num_layers,
                      source_sequence_length, source_vocab_size,
                      encoding_embedding_size):
    # Encoder embedding
    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)
    with tf.variable_scope("encoder"):
        # RNN cell
        cell = construct_cell(rnn_size, num_layers)

        encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input,
                                                          sequence_length=source_sequence_length, dtype=tf.float32)

    return encoder_output, encoder_state


# construct the decoder layer
def decoding_layer(target_letter_to_int, decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_length, max_target_sequence_length, encoder_state, decoder_input):
    # Embedding
    target_vocab_size = len(target_letter_to_int)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

    # construct cell
    cell = construct_cell(rnn_size, num_layers)

    # output fully connected layer
    output_layer = Dense(target_vocab_size,
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), name="dense_layer")

    # Training decoder
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
    # Predicting decoder
    # sharing parameters
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


# add <GO> and strip the last character for the input of the training decoder
def process_decoder_input(data, vocab_to_int, batch_size):
    ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return decoder_input


# connect the encoder and decoder
def seq2seq_model(input_data, targets, target_sequence_length,
                  max_target_sequence_length, source_sequence_length,
                  source_vocab_size,
                  encoder_embedding_size, decoding_embedding_size,
                  rnn_size, num_layers):
    _, encoder_state = get_encoder_layer(input_data,
                                         rnn_size,
                                         num_layers,
                                         source_sequence_length,
                                         source_vocab_size,
                                         encoder_embedding_size)

    decoder_input = process_decoder_input(targets, target_letter_to_int, batch_size)

    training_decoder_output, predicting_decoder_output, bm_decoder_output = decoding_layer(target_letter_to_int,
                                                                                           decoding_embedding_size,
                                                                                           num_layers,
                                                                                           rnn_size,
                                                                                           target_sequence_length,
                                                                                           max_target_sequence_length,
                                                                                           encoder_state,
                                                                                           decoder_input)

    return training_decoder_output, predicting_decoder_output, bm_decoder_output


# define the compute graph
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
    # get the logits to compute the loss
    training_logits = tf.identity(training_decoder_output.rnn_output, 'training_logits')
    predicting_logits = tf.identity(predicting_decoder_output.rnn_output, 'predicting_logits')
    # the result of the prediction
    prediction = tf.identity(predicting_decoder_output.sample_id, 'prediction_result')
    bm_prediction = tf.identity(bm_decoder_output.predicted_ids, 'bm_prediction_result')
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    # the score of the beam search prediction
    bm_score = tf.identity(bm_decoder_output.beam_search_decoder_output.scores, 'bm_prediction_scores')

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
        capped_gradients = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients, global_step=global_step)
    training_cost_summary = tf.summary.scalar('training_cost', train_cost)

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


# pad each batch to get the same size of input and output
def pad_sentence_batch(sentence_batch, pad_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


# generate batches
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


if isTrain:
    # shuffle the data set
    permu = np.random.permutation(len(source_int))
    source_int_shuffle = []
    target_int_shuffle = []

    for i in permu:
        source_int_shuffle.append(source_int[i])
        target_int_shuffle.append(target_int[i])

    train_source = source_int_shuffle[50 * batch_size:]
    train_target = target_int_shuffle[50 * batch_size:]

    valid_source = source_int_shuffle[:50 * batch_size]
    valid_target = target_int_shuffle[:50 * batch_size]


# calculate the error of the prediction
def cal_error(input_batch, predicition_result):
    t_error = 0.0
    for char_ids, pron_ids in zip(input_batch, predicition_result):
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


with tf.Session(graph=train_graph) as sess:
    # define summary file writer
    t_s = tf.summary.FileWriter('./graph/training', sess.graph)
    v_s = tf.summary.FileWriter('./graph/validation', sess.graph)
    # run initializer
    sess.run(tf.global_variables_initializer())

    # define saver, keep 15 most recent models
    saver = tf.train.Saver(max_to_keep=15)

    if isTrain:
        for epoch_i in range(1, epochs + 1):
            for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                    get_batches(train_target, train_source, batch_size,
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
                        get_batches(valid_target, valid_source, batch_size,
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

                    # calculate the validation cost and the WER over the validation dataset
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
        ckpt = tf.train.latest_checkpoint('./model')
        saver.restore(sess, ckpt)
        # convert the input data format
        while (True):
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
                for id in result[0][0, :, i]:
                    tmp.append(target_int_to_letter[id])
                    if id == target_letter_to_int['<EOS>']:
                        print ' '.join(tmp)
                        flag = 1
                        break
                # prediction length exceeds the max length
                if not flag:
                    print ' '.join(tmp)

                # print the score of the result
                print 'score: {0:.4f}'.format(result[1][0, :, i][-1])
                print ''
