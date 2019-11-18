#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = 'luheng'

import tensorflow as tf
from tensorflow import keras
from parameters import configs
from load_data import build_dataset
num_oov_buckets = configs.num_oov_buckets
embed_size = configs.embedding_size
n_units = configs.num_units
n_epochs = configs.n_epochs
n_outputs = configs.n_outputs
lr = configs.learning_rate
test_path = 'data/test.txt'
vocab_path = 'data/vocab.txt'
def vocab_func(vocab_path):
    file_vocab = open(vocab_path,'r')
    vocab = []
    for line in file_vocab:
        vocab.append(line.strip())
    file_vocab.close()
    return vocab
def input_data_func(data_path,shuffle=False):
    vocab = vocab_func(vocab_path)
    input_data = build_dataset(data_path)
    indices = tf.range(len(vocab), dtype=tf.int64)
    table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
    table = tf.lookup.StaticVocabularyTable(table_init,num_oov_buckets)
    def encode_words(X_batch, y_batch):
        return table.lookup(X_batch), y_batch
    input_data = input_data.map(encode_words).prefetch(1)
    return input_data
test_data = input_data_func(test_path)
print("begin load")
saved_model = keras.models.load_model("my_keras_model_check.h5")
print("end load begin evaluate!!!!!!!!!!!!!!!!!")
saved_model.evaluate(test_data)
y_pred = saved_model.predict(test_data)
print(y_pred)

