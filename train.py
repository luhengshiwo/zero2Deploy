#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = 'luheng'

import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from parameters import configs
from load_data import build_dataset

train_path = 'data/train.txt'
dev_path = 'data/dev.txt'
test_path = 'data/test.txt'
vocab_path = 'data/vocab.txt'
# embedding_path = 'data/word2vec.npy'
root_logdir = 'model/'
check_file_path = "my_keras_model_check.h5"
num_oov_buckets = configs.num_oov_buckets
embed_size = configs.embedding_size
n_units = configs.num_units
n_epochs = configs.n_epochs
n_outputs = configs.n_outputs
lr = configs.learning_rate

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

def matrix_func(embedding_path):
    matrix = np.load(embedding_path)
    return matrix

def vocab_func(vocab_path):
    file_vocab = open(vocab_path,'r')
    vocab = []
    for line in file_vocab:
        vocab.append(line.strip())
    file_vocab.close()
    return vocab

def input_data_func(data_path,shuffle=True):
    vocab = vocab_func(vocab_path)
    input_data = build_dataset(data_path)
    indices = tf.range(len(vocab), dtype=tf.int64)
    table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
    table = tf.lookup.StaticVocabularyTable(table_init,num_oov_buckets)
    def encode_words(X_batch, y_batch):
        return table.lookup(X_batch), y_batch
    input_data = input_data.map(encode_words).prefetch(1)
    return input_data

def train():
    vocab = vocab_func(vocab_path)
    train_data = input_data_func(train_path)
    dev_data = input_data_func(dev_path)
    test_data = input_data_func(test_path,shuffle=False)
    # matrix = matrix_func(embedding_path)
    run_logdir = get_run_logdir()
    model = keras.models.Sequential([
        keras.layers.Embedding(len(vocab)+num_oov_buckets, embed_size,
                            mask_zero=True,
                            # embeddings_initializer=tf.keras.initializers.Constant(matrix),
                            # trainable=True,
                            input_shape=[None]),
        keras.layers.GRU(n_units,return_sequences=True),
        keras.layers.GRU(n_units),
        keras.layers.Dense(n_outputs, activation="softmax")
    ])
    checkpoint_cb = keras.callbacks.ModelCheckpoint(check_file_path,
                                                save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=5,
                                                  restore_best_weights=True)
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    # tensorboard --logdir=./my_logs --port=6006
    optimizer = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999)
    loss = keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss, optimizer=optimizer,
                metrics=["accuracy"])
    if os.path.exists(check_file_path):
        model.load_weights(check_file_path)
        print("checkpoint_loaded......")
    history = model.fit(train_data, epochs=n_epochs,
                    validation_data=dev_data,
                    callbacks=[tensorboard_cb,checkpoint_cb,early_stopping_cb])
#     model.save('my_keras_model.h5')
    test_data = input_data_func(test_path)
    print('evaluate begins!!!!!!!!!!!!!!!!!!!')
    model.evaluate(test_data)
    # y_pred = model.predict(test_data)
    # print(y_pred)
    model_version = "0001"
    model_name = "my_cls_model"
    model_path = os.path.join(model_name, model_version)
    tf.saved_model.save(model, model_path)
#     x = [[1,4,6,7]]
#     saved_model = tf.saved_model.load(model_path)
#     y_pred = saved_model(x, training=False)
#     print(y_pred)


if __name__ == "__main__":
    train()