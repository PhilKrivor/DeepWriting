from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional
import numpy as np
import sys
import io
import os
import codecs
import re


#Variables
corpus = 'drseuss.txt'
MIN_WORD_FREQUENCY = 2
SEQUENCE_LEN = 5
dropout = 0.2
STEP = 1
BATCH_SIZE = 128
diversity = 1
seed = 'The sun did not'.lower()
sequence_length = 5
vocabulary_file = 'drseuss.txt'
quantity = 200
trainBool = False
#Extra Functions

def generator(sentence_list, next_word_list, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, SEQUENCE_LEN, len(words)), dtype=np.bool)
        y = np.zeros((batch_size, len(words)), dtype=np.bool)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index % len(sentence_list)]):
                x[i, t, word_indices[w]] = 1
            y[i, word_indices[next_word_list[index % len(sentence_list)]]] = 1
            index = index + 1
        yield x, y

def shuffle_and_split_training_set(sentences_original, next_original, percentage_test=2):
    # shuffle at unison
    print('Shuffling sentences')

    tmp_sentences = []
    tmp_next_word = []
    for i in np.random.permutation(len(sentences_original)):
        tmp_sentences.append(sentences_original[i])
        tmp_next_word.append(next_original[i])

    cut_index = int(len(sentences_original) * (1.-(percentage_test/100.)))
    x_train, x_test = tmp_sentences[:cut_index], tmp_sentences[cut_index:]
    y_train, y_test = tmp_next_word[:cut_index], tmp_next_word[cut_index:]

    print("Size of training set = %d" % len(x_train))
    print("Size of test set = %d" % len(y_test))
    return (x_train, y_train), (x_test, y_test)

def print_vocabulary(words_file_path, words_set):
    words_file = codecs.open(words_file_path, 'w', encoding='utf8')
    for w in words_set:
        if w != "\n":
            words_file.write(w+"\n")
        else:
            words_file.write(w)
    words_file.close()


def generate_text(model, indices_word, word_indices, seed,
                  sequence_length, diversity, quantity):
    """
    Similar to lstm_train::on_epoch_end
    Used to generate text using a trained model
    :param model: the trained Keras model (with model.load)
    :param indices_word: a dictionary pointing to the words
    :param seed: a string to be used as seed (already validated and padded)
    :param sequence_length: how many words are given to the model to generate
    :param diversity: is the "temperature" of the sample function (usually between 0.1 and 2)
    :param quantity: quantity of words to generate
    :return: Nothing, for now only writes the text to console
    """
    sentence = seed.split(" ")
    print("----- Generating text")
    print('----- Diversity:' + str(diversity))
    print('----- Generating with seed:\n"' + seed)

    print(seed)
    for i in range(quantity):
        x_pred = np.zeros((1, sequence_length, len(vocabulary)))
        for t, word in enumerate(sentence):
            x_pred[0, t, word_indices[word]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_word = indices_word[next_index]

        sentence = sentence[1:]
        sentence.append(next_word)

        print(" "+next_word, end="")
    print("\n")

def validate_seed(vocabulary, seed):
    """Validate that all the words in the seed are part of the vocabulary"""
    print("\nValidating that all the words in the seed are part of the vocabulary: ")
    seed_words = seed.split(" ")
    valid = True
    for w in seed_words:
        print(w, end="")
        if w in vocabulary:
            print(" ✓ in vocabulary")
        else:
            print(" ✗ NOT in vocabulary")
            valid = False
    return valid

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

#--------------------

with io.open(corpus, encoding='utf-8') as f:
    text = f.read().lower().replace('\n', ' \n ')
print('Corpus length in characters:', len(text))

text_in_words = [w for w in text.split(' ') if w.strip() != '' or w == '\n']

print('Corpus length in words:', len(text_in_words))

text_in_words = text_in_words



word_freq = {}
for word in text_in_words:
    word_freq[word] = word_freq.get(word, 0) + 1

ignored_words = set()
for k, v in word_freq.items():
    if word_freq[k] < MIN_WORD_FREQUENCY:
        ignored_words.add(k)

words = set(text_in_words)
print('Unique words before ignoring:', len(words))
print('Ignoring words with frequency <', MIN_WORD_FREQUENCY)
words = sorted(set(words) - ignored_words)
print('Unique words after ignoring:', len(words))

word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))

# cut the text in semi-redundant sequences of SEQUENCE_LEN words
STEP = 1
sentences = []
next_words = []
ignored = 0
for i in range(0, len(text_in_words) - SEQUENCE_LEN, STEP):
    # Only add sequences where no word is in ignored_words
    if len(set(text_in_words[i: i+SEQUENCE_LEN+1]).intersection(ignored_words)) == 0:
        sentences.append(text_in_words[i: i + SEQUENCE_LEN])
        next_words.append(text_in_words[i + SEQUENCE_LEN])
    else:
        ignored = ignored+1
print('Ignored sequences:', ignored)
print('Remaining sequences:', len(sentences))



(sentences, next_words), (sentences_test, next_words_test) = shuffle_and_split_training_set(sentences, next_words)

model = Sequential()
model.add(Bidirectional(LSTM(128), input_shape=(SEQUENCE_LEN, len(words))))
if dropout > 0:
    model.add(Dropout(dropout))
model.add(Dense(len(words)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

filepath="DeepLearningV3Weights/weights-improvement-{epoch:02d}-{loss:.4f}-seuss.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

if trainBool == True:

    model.fit_generator(generator(sentences, next_words, BATCH_SIZE),
                            steps_per_epoch=int(len(sentences)/BATCH_SIZE) + 1,
                            epochs=100,
                            callbacks=callbacks_list,
                            validation_data=generator(sentences_test, next_words_test, BATCH_SIZE),
                            validation_steps=int(len(sentences_test)/BATCH_SIZE) + 1)
else:

    model.summary()
    model.load_weights('DeepLearningV3Weights/test3.hdf5')
    #vocabulary = open(vocabulary_file, "r").readlines()
    # remove the \n at the end of the word, except for the \n word itself
    #vocabulary = [re.sub(r'(\S+)\s+', r'\1', w) for w in vocabulary]
    #vocabulary = sorted(set(vocabulary))
    vocabulary = words

    word_indices = dict((c, i) for i, c in enumerate(vocabulary))
    indices_word = dict((i, c) for i, c in enumerate(vocabulary))

    if validate_seed(vocabulary, seed):
        print("\nSeed is correct.\n")
        # repeat the seed in case is not long enough, and take only the last elements
        seed = " ".join((((seed+" ")*sequence_length)+seed).split(" ")[-sequence_length:])
        generate_text(
            model, indices_word, word_indices, seed, sequence_length, diversity, quantity
        )
    else:
        print('\033[91mERROR: Please fix the seed string\033[0m')
        exit(0)


