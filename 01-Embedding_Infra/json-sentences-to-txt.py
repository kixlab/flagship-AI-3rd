# -*- coding: utf-8 -*-

import os
import json

SUB_DIRECTORY = 'dataset/sci-news-sum-kr-50/data/'

dirname = os.path.dirname(__file__)

document = []
for i in (range(1, 51)):
  filename = os.path.join(dirname, SUB_DIRECTORY + "%02d" % i + '.json')
  with open(filename, encoding="utf-8") as json_file:
    json_data = json.loads(json_file.read())
    document.extend(json_data['sentences'])

with open(os.path.join(dirname, 'sci-total-sentences.txt'), 'w') as f:
  for sentence in document:
    f.write(sentence + os.linesep)
