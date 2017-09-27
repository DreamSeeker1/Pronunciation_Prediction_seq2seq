import pickle

# open the file contains data
with open('/home/yy/pronunciation-prediction/tensor_seq/source_list') as f:
    source_data = f.read().split('\n')
with open('/home/yy/pronunciation-prediction/tensor_seq/target_list') as f:
    target_data = f.read().split('\n')


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

with open('data.pickle', 'w') as f:
    pickle.dump([source_int_to_letter, source_letter_to_int, target_int_to_letter, target_letter_to_int, source_int,
                 target_int], f)
