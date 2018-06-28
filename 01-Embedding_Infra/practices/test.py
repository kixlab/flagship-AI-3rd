# -*- coding: utf-8 -*-

import os

dirname = os.path.dirname(__file__)
filename = os.path.join(os.path.join(dirname, os.pardir), 'dataset/sci-total-sentences.txt')

with open(filename) as f:
  data = f.read().strip()
  sentences = data.split(os.linesep)
print(sentences)
# import sys

# print(sys.stdin.encoding)