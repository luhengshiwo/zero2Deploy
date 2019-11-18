#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = 'luheng'

import sys
import tensorflow as tf
import time
import os
from parameters import configs

def build_dataset(path,shuffle=True):
    batch_size = configs.batch_size
    pad_value = configs.pad
    dataset = tf.data.TextLineDataset(path)
    dataset = dataset.map(
        lambda line: (tf.compat.v1.string_split([line]).values[1:], tf.compat.v1.string_split([line]).values[0]),
        num_parallel_calls=configs.num_parallel_calls
    )
    dataset = dataset.map(lambda text,label: (text,tf.strings.to_number(
        label,
        out_type=tf.dtypes.int64,
        name=None
    )))
    if shuffle == True:
        dataset = dataset.shuffle(buffer_size=1000, seed=42)
    dataset = dataset.padded_batch(batch_size,padded_shapes=([-1],[]),padding_values=(tf.constant(pad_value, dtype=tf.string),tf.constant(1, dtype=tf.int64)))
    return dataset

if __name__ == "__main__":
    path = 'data/dev.txt'
    dataset = build_dataset(path)
    for i in dataset.take(1):
        print(i)