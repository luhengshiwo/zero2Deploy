#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = 'luheng'

import time
import os
import subprocess
origin_path = 'data/origin.txt'
cut_path = 'data/cut.txt'
embedding_path = 'data/for_embedding.txt'

def normal_cut(origin_path, cut_path,embedding_path):
    train_file = open(origin_path)
    cut_file = open(cut_path, 'w+')
    embedding_file= open(embedding_path, 'w+')
    for line in train_file:
        lines = line.strip().split('|')
        text = " ".join(list(lines[1]))
        label = lines[0]
        cut_file.write(label+' '+text+'\n')
        embedding_file.write(text+'\n')
    train_file.close()
    cut_file.close()
    embedding_file.close()

def bash_shell():
    shell_path = 'train_dev_test_split.sh'
    subprocess.call(['bash', shell_path])

def main():
    normal_cut(origin_path, cut_path,embedding_path)
    bash_shell()

if __name__ == '__main__':
    tic = time.time()
    main()
    tok = time.time()
    cost = tok-tic
    print('cost time:{:.2f}'.format(cost))