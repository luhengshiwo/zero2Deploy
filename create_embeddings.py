#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = 'luheng'

import time
import numpy as np
import os
from gensim.models import word2vec
from parameters import configs


'''
1.接收一个文件，path
2.输出3个文件，训练集，字典，wordvec向量
'''
embedding_size = configs.embedding_size
pad = configs.pad

train_path = 'data/for_embedding.txt'
vocab_path = 'data/vocab.txt'
vec_path = 'data/word2vec.npy'
model_path = 'data/source'

def data_transform(origin_file,for_train_path,embedding_path):
    origin = open(origin_file)
    train = open(for_train_path,'w+')
    embedding = open(embedding_path,'w+')
    #do something of the origin data
    origin.close()
    train.close()
    embedding.close()

def create_embeddings(filepath, modelpath, vocab, vec):
    sentences = word2vec.Text8Corpus(filepath)
    model = word2vec.Word2Vec(
        sentences, size=embedding_size, min_count=0, max_vocab_size=100000)
    model.save(modelpath)
    vocab_f = open(vocab, 'w+')
    vocab_f.write(pad + '\n')
    model = word2vec.Word2Vec.load(modelpath)
    all_words = set()
    vectors = []
    random_vec = (-1 + 2 *np.random.random(embedding_size)).astype(np.float32)
    vectors.append(random_vec)
    for line in open(filepath):
        words = line.split(" ")
        for word in words:
            word = word.strip()
            if word != '':
                all_words.add(word)
    for word in all_words:
        try:
            vector = model[word]
            vectors.append(vector)
            vocab_f.writelines(word + '\n')
        except:
            pass
    vectors.append(random_vec)
    vocab_f.close()
    np.save(vec,np.array(vectors))
    os.remove(modelpath)

def main():
    create_embeddings(train_path, model_path, vocab_path, vec_path)

if __name__ == '__main__':
    tic = time.time()
    main()
    tok = time.time()
    cost = tok-tic
    print('cost time:{:.2f}'.format(cost))
