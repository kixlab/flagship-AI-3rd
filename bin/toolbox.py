# -*- coding: utf-8 -*-

import os
import json
import math
import universal_utils as uu

## Varaibles
directory = '../results/token-scripts-reduce3'
log_per = 10000
# input_file = '../results/token-scripts-reduce3-words.json'
# output_file = '../results/token-scripts-reduce3-word-list.txt'
words_file = '../results/token-scripts-reduce3-word-list.txt'
sentences_file = '../results/token-scripts-reduce3.txt'
output_file = '../results/token-scripts-reduce3-skipgram.txt'

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

# to get the number of frequency of words with json file
# text file of sentences of tokens (able to be splited by whitespace) -> json file of frequency
def get_word_frequency_json(input_fn, output_fn, l_p):
  print("[%s] Import file..." % uu.get_current_datetime())
  with open(input_fn, 'rb') as readfile:
    sentences = [l.decode('utf8', 'ignore')
                  for l in readfile.readlines()]
    
    print("[%s] Start to count the number of word's presence" % uu.get_current_datetime())
    result = {}
    for idx, s in enumerate(sentences, 1):
      for w in s.strip().split(' '):
        if (w == ''):
          continue
        
        if (w in result):
          result[w] += 1
        else:
          result[w] = 1
      if (idx % l_p) == 0:
        print("[%s] %7dth sentence was read; total %7d words" % (uu.get_current_datetime(), idx, len(result.keys())))
    
    print("[%s] total %7d sentences were read; total %7d words" % (uu.get_current_datetime(), len(sentences), len(result.keys())))
  
  with open(output_fn, 'w') as writefile:
    print("[%s] writing result on json file..." % uu.get_current_datetime())
    json.dump(result, writefile, ensure_ascii=False)
  
  print("[%s] All process completed!" % uu.get_current_datetime())


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

def show_counts_log_scale(filename, scale = 2, min_frequency = 5):
  print('[%s] Start process...' % uu.get_current_datetime())
  with open(filename, 'r') as readfile:
    counts = json.load(readfile)
    result = {}
    for k in counts.keys():
      log_value = int(math.log(int(counts[k]), scale))
      if log_value in result:
        result[log_value] += 1
      else:
        result[log_value] = 1
  
  print('[%s] Process completed!' % uu.get_current_datetime())
  able_words_num = 0
  for k in sorted(result.keys()):
    print('Scale %2d: %7d' % (k, result[k]))
    if (k >= math.log(min_frequency, scale)):
      able_words_num += result[k]

  print('Total %d of %d words can be used for word2vec! (%f%%)' % (able_words_num, len(counts), (able_words_num / len(counts) * 100)))


def create_word_list(words_fn, output_fn, min_frequency = 5, log_per = 100000):
  word_list = []   # Words present over min_frequency

  # Load freuqent words list
  print('[%s] Load words...' % uu.get_current_datetime())
  with open(words_fn, 'r') as read_words:
    word_dict = json.load(read_words)
    for k in word_dict.keys():
      if int(word_dict[k]) >= min_frequency:
        word_list.append(k)
  print('[%s] Complete to load words; total %d of %d words' % (uu.get_current_datetime(), len(word_list), len(word_dict)))
  
  uu.print_dt('Saving word_list into %s ...' % output_fn)
  with open(output_fn, 'w') as writefile:
    for w in word_list:
      writefile.write(w + os.linesep)

def print_rare_words(counts_fn, max_frequency = 2, print_num = 20):
  with open(counts_fn, 'r') as readfile:
    word_dict = json.load(readfile)
    
    printed_num = 0
    for k in word_dict.keys():
      if (int(word_dict[k]) <= max_frequency):
        print("%s (%s times)" % (k, word_dict[k]))
        printed_num += 1

      if (printed_num >= print_num):
        return


def create_skip_grams(word_list_fn, sentences_fn, output_fn, window_size=2, log_per=100000):
  # Load sentences
  uu.print_dt('Load sentences...')
  sentences = uu.load_text_file(sentences_fn)

  # Load word list
  uu.print_dt('Load word list...')
  word_list = uu.load_text_file(word_list_fn)
  word_dict = {w: i for i, w in enumerate(word_list)}

  # Create skip_grams
  uu.print_dt('Create skip_grams...')
  skip_grams = []
  all_skip_grams = 0
  for idx, s in enumerate(sentences[:100000], 1):
    words_in_sentences = s.strip().split(' ')
    for w_idx, w in enumerate(words_in_sentences, 0):
      target = w
      # Only add number of sets to all_skip_grams
      if (target not in word_list):
        all_skip_grams += min([window_size, w_idx]) + \
            min([window_size, len(words_in_sentences) - w_idx - 1])
        continue

      # Create word set
      for c_idx in range(w_idx - window_size, w_idx + window_size + 1):
        if (c_idx == w_idx):
          continue

        if (c_idx < 0 or c_idx >= len(words_in_sentences)):
          continue
        
        all_skip_grams += 1
        content = words_in_sentences[c_idx]
        if content in word_list:
          skip_grams.append([word_dict[target], word_dict[content]])
    if (idx % log_per == 0):
      print_set = (idx, len(skip_grams), all_skip_grams, len(skip_grams) / all_skip_grams * 100)
      uu.print_dt("%7d sentences were parsed: %8d of %8d skip-grams can be used. (%2f%%)" % print_set)

  print_set = (len(skip_grams), all_skip_grams, len(skip_grams) / all_skip_grams * 100)
  uu.print_dt("All sentences were parsed: %8d of %8d skip-grams can be used. (%2f%%)" % print_set)

  uu.print_dt("Save skip-grams...")
  with open(output_fn, 'w') as writefile:
    for s_g in skip_grams:
      writefile.write("%d %d" % (s_g[0], s_g[1]) + os.linesep)

## Main
# create_word_list(input_file, output_file)
create_skip_grams(words_file, sentences_file, output_file, log_per=log_per)