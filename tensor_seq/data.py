import pickle
# open the file contains data
with open('./source_list') as f:
    source_data = f.read().split('\n')
with open('./target_list') as f:
    target_data = f.read().split('\n')

# map the word with the pronunciation
word_pron = {}
for word, pron in zip(source_data, target_data):
    if word in word_pron:
        word_pron[word].add(pron)
    else:
        word_pron[word] = {pron}

# map the character and pronunciation to idx
def word2index(data):
    """Map each character in the word list and special symbols to idx(int)
    Args:
        data: a list contains the words.
    Returns:
        idx2word: a dict map idx to character and symbol.
        word2idx: a dict map idx to character and symbol.
    """
    special_words = ['<EOS>', '<GO>', '<PAD>', '<UNK>']
    set_words = list(set([c for line in data for c in line] + special_words))

    idx2word = {idx: word for idx, word in enumerate(set_words)}
    word2idx = {word: idx for idx, word in idx2word.items()}
    return idx2word, word2idx


def phon2index(data):
    """Map each symbol in arpabet and special symbol to idx
    Args:
        data: a list contains the pronunciation sequence for each word.
    Returns:
        idx2word: a dict map each idx to a symbol.
        word2idx: a dict map each symbol to an idx.
    """
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

# save all the objects to a file.
with open('data.pickle', 'w') as f:
    pickle.dump([source_int_to_letter, source_letter_to_int, target_int_to_letter, target_letter_to_int, source_int,
                 target_int, word_pron], f)
