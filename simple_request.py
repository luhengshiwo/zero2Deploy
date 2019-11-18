#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = 'luheng'
import argparse
import requests
import json

API_ENDPOINT = "http://localhost:5000/cls/"
sentence = '这个手机很垃圾'
r = requests.post(url=API_ENDPOINT,data=sentence.encode('utf-8'))
print(r.text)
