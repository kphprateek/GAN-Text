import pickle
import re
import string

import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.tokenize import word_tokenize, sent_tokenize

from settings import *


def clean_text(text):
    TAG_RE = re.compile(r'<[^>]+>')

    def tokenize_text(text):
        return [w for s in sent_tokenize(text) for w in word_tokenize(s)]

    def remove_special_characters(text, characters=string.punctuation.replace('-', '')):
        tokens = tokenize_text(text)
        pattern = re.compile('[{}]'.format(re.escape(characters)))
        return ' '.join(filter(None, [pattern.sub('', t) for t in tokens]))

    def remove_tags(text):
        return TAG_RE.sub('', text)

    text = text.strip(' ')
    text = remove_tags(text)
    text = text.lower()
    text = remove_special_characters(text)
    return text


def split_and_clean(data):
    arr = []
    for i in data:
        sentences = i.split('.')
        for s in sentences:
            if MIN_SEQ_LENGTH < len(s.split()) <= SEQ_LENGTH:
                if s:
                    arr.append(s)
    for i, j in enumerate(arr):
        arr[i] = clean_text(arr[i])
    return arr


def dump_tokenizer(tokenizer):
    with open(tokenizer_file, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_tokenizer():
    with open(tokenizer_file, 'rb') as handle:
        tokenizer = pickle.load(handle)

    return tokenizer


def tokenize(arr):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', num_words=vocab_size)
    tokenizer.fit_on_texts(arr)
    tensor = tokenizer.texts_to_sequences(arr)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post', maxlen=SEQ_LENGTH)
    dump_tokenizer(tokenizer)
    return tensor


def dump_tokenizer_tag(tokenizer):
    with open(tokenizer_file_tag, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_tokenizer_tag():
    with open(tokenizer_file_tag, 'rb') as handle:
        tokenizer = pickle.load(handle)

    return tokenizer


def tokenize_tag(arr):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', num_words=vocab_size)
    tokenizer.fit_on_texts(arr)
    tensor = tokenizer.texts_to_sequences(arr)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post', maxlen=SEQ_LENGTH)
    dump_tokenizer_tag(tokenizer)
    return tensor


if __name__ == "__main__":
    # global d_tag
    df = pd.read_csv(dataset_path)
    df_tag = pd.read_csv('C:/Users/nbtc068/Desktop/seqgan-text-generation-tf2/dataset/coco_tag.csv')
    df = df.review[:generated_num]
    df_tag=df_tag.review[:generated_num]
    d = split_and_clean(df)
    d_tag=split_and_clean(df_tag)
    d = tokenize(d)
    d_tag=tokenize_tag(d_tag)

    np.savetxt('dataset/positives.txt', d[:generated_num], delimiter=' ', fmt='%i')
    np.savetxt('dataset/negatives.txt', d[generated_num:(2 * generated_num)], delimiter=' ', fmt='%i')
    np.savetxt('dataset/positives_tag.txt', d_tag[:generated_num], delimiter=' ', fmt='%i')
    np.savetxt('dataset/negatives_tag.txt', d_tag[generated_num:(2 * generated_num)], delimiter=' ', fmt='%i')
