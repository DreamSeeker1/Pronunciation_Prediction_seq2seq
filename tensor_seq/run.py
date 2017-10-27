"""Train or test the model"""

from model import *

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

    # define saver, keep max_model_number of most recent models
    saver = tf.train.Saver(max_to_keep=max_model_number)

    if isTrain == 1:
        # run initializer
        sess.run(tf.global_variables_initializer())

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
