# -*- coding: utf-8 -*-

import os
import json

## Varaibles
directory = '../results/token-scripts-plain'
log_per = 200


## Functions
def convert_scripts(direc, log_per):
  with open(directory + '.txt', 'w') as writefile:
    for idx, f in enumerate(os.listdir(direc), 1):
      if (f.endswith('.json')):
        with open(os.path.join(direc, f), 'r') as readfile:
          dialogs = json.load(readfile)
          for d in dialogs:
            for l in d['lines']:
              if (len(l['tokens']) > 0):
               writefile.write(' '.join(l['tokens']) + os.linesep)
      if (idx % log_per == 0):
        print("Converting step %6d" % idx)

def cut_sentences(filename, line_num):
  with open(filename, 'r') as readfile:
    sentences = readfile.read().strip().split(os.linesep)
    filename_pure, extension = os.path.splitext(filename)
    with open(filename_pure + '-%d' % line_num + extension, 'w') as writefile:
      for s in sentences[:line_num]:
        writefile.write(s + os.linesep)

def convert_sci_news():
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

## Main
# convert_scripts(directory, log_per)
cut_sentences('../results/token-scripts-plain.txt', 100000)