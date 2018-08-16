# -*- coding: utf-8 -*-

import os
import json
import math
import time
import re
import universal_utils as uu
from tqdm import trange, tqdm
from random import shuffle
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np

## Varaibles
# directory = '../results/script_filtered.txt'
vrm_directory = '../scripts/VRMtraining'
# log_per = 10000
input_file = '../results/vrm_data_fixed_final.json'
# output_file = '../results/vrm_data_fixed_final.json'
# split_output_file = '../results/lstm-testset/ne.txt'
# words_file = '../results/180731-padded-500000sent_final.json'
# words_file = '../results/token-scripts-reduce3-word-list.txt'
# sentences_file = '../results/token-scripts-reduce3.txt'
# output_file = '../results/token-scripts-reduce3-skipgram.txt'
log_file = '../logs/toolbox.log'

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

def count_sentences_in_jsons(direc, log_file):
  logger = uu.get_custom_logger('count_sentences', log_file)
  logger.info(f"Count sentences in {direc} directory...")
  total_lines = 0
  for f in os.listdir(direc):
    if (f.endswith('.json')):
      with open(os.path.join(direc, f), 'r') as readfile:
        dialogs = json.load(readfile)
        for d in dialogs:
          for l in d['lines']:
            if (len(l['message']) > 0):
              total_lines += 1
  logger.info(f"Total sentences in {direc}: {total_lines}")
              
def convert_script_json_txt(direc, output_fn, log_fn):
  logger = uu.get_custom_logger('convert_json_txt', log_fn)
  start_time = time.time()
  with open(output_fn, 'w') as writefile:
    for f in tqdm(os.listdir(direc)):
      if (f.endswith('.json')):
        with open(os.path.join(direc, f), 'r') as readfile:
          dialogs = json.load(readfile)
          for d in dialogs:
            for l in d['lines']:
              content = f"[[{l['speaker']}]] {l['message']}"
              writefile.write(content + os.linesep)
          writefile.write(os.linesep)

  end_time = time.time() - start_time
  logger.info("Finish to convert script json to txt file: %.2f sec" % end_time)

def cut_sentences(filename, line_num):
  with open(filename, 'r') as readfile:
    sentences = readfile.read().strip().split(os.linesep)
    filename_pure, extension = os.path.splitext(filename)
    with open(filename_pure + '-%d' % line_num + extension, 'w') as writefile:
      for s in sentences[:line_num]:
        writefile.write(s + os.linesep)

# to get the number of frequency of words with json file
# text file of sentences of tokens (able to be splited by whitespace) -> json file of frequency
def get_word_frequency_json(input_fn, output_fn, l_p, max_words=500000):
  print("[%s] Import file..." % uu.get_current_datetime())
  with open(input_fn, 'rb') as readfile:
    sentences = [l.decode('utf8', 'ignore')
                  for l in readfile.readlines()][:max_words]
    
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

def split_sentences_in_txt(input_fn, output_fn, log_fn):
  ELLIPSIS_RE = re.compile('\.\.+|…')
  IN_BRACKETS_RE = re.compile('\(.*?\)')  # Limitation on nested brackets like '(a(b)c)'

  logger = uu.get_custom_logger('toolbox', log_fn)
  sentences = uu.load_text_file(input_fn)
  results = []
  logger.info('Split sentences...')
  for s in tqdm(sentences):
    s = s.strip()
    if len(s) == 0 or s == '' or not s.startswith('[['):
      results.append('')
    else:
      if ' ' not in s:
        continue
      result = []
      speaker = s.split(' ')[0]
      replaced_s = IN_BRACKETS_RE.sub('', ' '.join(s.split(' ')[1:]))
      replaced_s = ELLIPSIS_RE.sub(' ', replaced_s)
      splited_s = ''
      for w in replaced_s.strip():
        if w == '.':
          if len(splited_s) > 0:
            result.append(speaker + ' ' + splited_s)
            splited_s = ''
        elif w == '!' or w == '?':
          result.append(speaker + ' ' + splited_s + w)
          splited_s = ''
        else:
          splited_s += w
      if len(splited_s) > 0:
        result.append(speaker + ' ' + splited_s)
      results.extend(result)
  
  logger.info('Save results...')
  with open(output_fn, 'w') as writefile:
    for r in tqdm(results):
      writefile.write(r + os.linesep)
  logger.info(f'Done - {len(sentences)} sentences => {len(results)} sentences')

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

def get_info_of_sentences(sentences_fn, sentences_num):
  if sentences_num < 1:
    print("ERROR: sentences_num MUST be more than 1.")
    return

  print("Start to read file...")
  sentences = uu.load_text_file(sentences_fn)[:sentences_num]
  single_word_sentence = 0
  total_words = []

  print("Get words from sentences...")
  for s in sentences:
    words = s.strip().split(' ')
    if len(words) == 1:
      single_word_sentence += 1
    total_words.extend(words)
  
  total_words_num = len(total_words)
  unique_words_num = len(list(set(total_words)))

  logger = uu.get_custom_logger('sentences_info', os.path.join(uu.get_base_path(), 'logs/sentences_info.log'))
  logger.info(f'{sentences_num} sentences INFO:')
  logger.info('Total words: %d | Unique words: %d (%.2f%% of total)' % (
    total_words_num, unique_words_num, unique_words_num / total_words_num * 100))
  logger.info('Words per sentences: %.2f' % (total_words_num / sentences_num))
  logger.info("Single-word-sentences: %d (%.2f%% of total)" % (
    single_word_sentence, single_word_sentence / sentences_num * 100))
  logger.info("=" * 50)

def make_omitted_sentences(sentences_fn, output_fn, sentences_num, min_count):
  if sentences_num < 1:
    print("ERROR: sentences_num MUST be more than 1.")
    return

  print("Start to read file...")
  sentences = uu.load_text_file(sentences_fn)[:sentences_num]

  print("Get word_counts from sentences...")
  word_counts = {}
  for s in tqdm(sentences):
    words = s.strip().split(' ')
    for w in words:
      if w in word_counts.keys():
        word_counts[w] += 1
      else:
        word_counts[w] = 1
  
  print("Get frequent words list...")
  frequent_words = []
  for k in tqdm(word_counts.keys()):
    if word_counts[k] >= min_count:
      frequent_words.append(k)
  logger = uu.get_custom_logger('info_omitted', os.path.join(uu.get_base_path(), 'logs/omit.log'))
  logger.info("Omitting ~%d Sentences with min_count %d" % (sentences_num, min_count))
  frequent_len = len(frequent_words)
  total_len = len(word_counts)
  logger.info("Survived Vocabs: %d of Total %d (%.2f%%)" % (frequent_len, total_len, frequent_len / total_len * 100))

  print("Write results...")
  total_words_len = 0
  omitted_words_len = 0
  with open(output_fn, 'w') as writefile:
    for s in tqdm(sentences):
      words = s.strip().split(' ')
      omitted_words = []
      for idx, w in enumerate(words):
        if w not in frequent_words:
          words[idx] = '()'
          omitted_words.append(w)
      omitted_words_len += len(omitted_words)
      total_words_len += len(words) - omitted_words_len
      writefile.write("%s [%s]" % (' '.join(words), ', '.join(omitted_words))
                      + os.linesep)
  frequent_words_len = total_words_len - omitted_words_len
  logger.info("Survived Words: %d of Total %d (%.2f%%)"
                % (frequent_words_len, total_words_len, frequent_words_len / total_words_len * 100))
  logger.info("-" * 50)

def read_word_dict_json(words_fn, logger):
  logger.info("Reading word_dict file...")
  with open(words_fn, encoding="utf-8") as json_file:
    word_dict = json.loads(json_file.read())
  word_dict = sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)
  logger.info("Complete to read!")
  return word_dict

def select_with_word_txt(sentences_fn, output_fn, target_word, logger, max_len=10000):
  logger.info('Start processing...')
  sentences = uu.load_text_file(sentences_fn)
  shuffle(sentences)

  positive_ss = []
  negative_ss = []
  for s in tqdm(sentences):
    if s == '':
      continue
    is_positive = False
    for w in s.split(' '):
      if w == target_word:
        if len(positive_ss) < max_len:
          positive_ss.append(s)
        is_positive = True
        break
    if not is_positive and len(negative_ss) < max_len:
      negative_ss.append(s)
    if len(positive_ss) >= max_len and len(negative_ss) >= max_len:
      break

  logger.info("Write results...")
  filename, file_extension = os.path.splitext(output_fn)
  with open(f"{filename}_positive{file_extension}", 'w') as writefile:
    writefile.write(os.linesep.join(positive_ss))
  with open(f"{filename}_negative{file_extension}", 'w') as writefile:
    writefile.write(os.linesep.join(negative_ss))

# Warning: Too slow!
def remove_less_frequent_words(sentences_fn, words_fn, output_fn, logger, frequent_num=5):
  sentences = uu.load_text_file(sentences_fn)
  with open(words_fn, 'r') as readfile:
    words_dict = json.load(readfile)
  words = list(filter(lambda k: words_dict[k] >= 5, words_dict))
  new_sentences = []
  for s in tqdm(sentences):
    new_words = []
    for w in s.split(' '):
      if w == '0' or w in words:
        new_words.append(w)
      else:
        ['0'] + new_words
    new_sentences.append(' '.join(new_words))
  
  with open(output_fn, 'w') as writefile:
    writefile.write(os.linesep.join(new_sentences))

def adjust_sentence_len(sentences_fn, output_fn, sentence_len=16):
  sentences = uu.load_text_file(sentences_fn)
  new_sentences = []
  for s in tqdm(sentences):
    words = s.split(' ')
    if len(words) > sentence_len:
      words = words[:sentence_len]
    elif len(words) < sentence_len:
      words = ['0'] * (sentence_len-len(words)) + words
    new_sentences.append(' '.join(words))
  with open(output_fn, 'w') as writefile:
    writefile.write(os.linesep.join(new_sentences))

def extract_script_from_vrm(vrm_direc, output_fn):
  outputs = []
  for f in tqdm(os.listdir(vrm_direc)):
    if (f.endswith('.VRM')):
      with open(os.path.join(vrm_direc, f), 'r') as readfile:
        contents = readfile.read().split('FF')
        if len(contents) >= 3:
          outputs.extend(contents[1:-1])
  with open(output_fn, 'w') as writefile:
    writefile.write((os.linesep + os.linesep).join(outputs))


def revise_script_from_vrm(input_fn, output_fn, logger):
  with open(input_fn, 'r') as readfile:
    contents = readfile.read().split(os.linesep)
    results = []
    for c in contents:
      if (not (c.endswith('UU') or c.endswith('MM'))):
        results.append(c)
  with open(output_fn, 'w') as writefile:
    writefile.write(os.linesep.join(results))
  logger.info(f"{len(contents) - len(results)} of {len(contents)} are removed!")

def vrm_script_to_json(input_fn, output_fn, logger):
  in_brackets_re = re.compile('\(.*?\)')

  sentences = uu.load_text_file(input_fn)
  dialogs = []
  speechs = []
  speakers = []
  speaker = 'A'

  for s in tqdm(sentences):
    s = s.strip()
    if s == '':
      # If blank line, push speechs into dialogs & reset variables
      if len(speechs) > 0:
        dialogs.append(speechs)
      speechs = []
      speakers = []
      speaker = 'A'
    else:
      # Remove words in parentheses
      s = in_brackets_re.sub(' ', s)

      # Remove colons between numbers
      s = re.sub('([0-9]+[:][0-9]+)', (lambda obj: obj.string.replace(':', ' ')), s)

      # Split content and VRM tag
      content, vrm = s[:-2].strip(), s[-2:]

      # If there is info for speaker, normalize it like 'A', 'B', ...
      if ':' in content:
        raw_speaker, content = content.split(':')[0].strip(), content.split(':')[1].strip()
        if raw_speaker not in speakers:
          speakers.append(raw_speaker)
        speaker = chr(65 + speakers.index(raw_speaker))
      
      # Save speech into speechs
      speechs.append({
        'speaker': speaker,
        'utterance': content,
        'vrm': vrm
      })
  
  with open(output_fn, 'w') as writefile:
    json.dump(dialogs, writefile)

def remove_duplicates_vrm_json(input_fn, output_fn, logger):
  with open(input_fn, 'r') as readfile:
    dialogs = json.load(readfile)
    no_dup = []
    possible_dup = []
  for d in dialogs:
    if len(d) > 1:
      no_dup.append(d)
    else:
      possible_dup.extend(d)
  removed_dup = [[dict(t)] for t in {tuple(d.items()) for d in possible_dup}]
  logger.info(f"Remove duplicates in vrm.json: {len(possible_dup)} > {len(removed_dup)}")

  with open(output_fn, 'w') as writefile:
    json.dump(no_dup + removed_dup, writefile)

def test_integrity_vrm_json(input_fn, logger):
  vrm_category = ['D', 'E', 'A', 'C', 'Q', 'K', 'I', 'R', 'U']

  with open(input_fn, 'r') as readfile:
    dialogs = json.load(readfile)
  invalid_vrm_num = 0
  invalid_speaker_num = 0
  for d in dialogs:
    for u in d:
      if (u['speaker'] not in ['A', 'B', 'C']):
        invalid_speaker_num += 1
      if (len(u['vrm']) != 2 or u['vrm'][0] not in vrm_category or u['vrm'][1] not in vrm_category):
        invalid_vrm_num += 1
  logger.info(f'Invalid speaker: {invalid_speaker_num}, Invalid VRM: {invalid_vrm_num}')

def count_intent_vrm_json(input_fn, logger):
  with open(input_fn, 'r') as readfile:
    dialogs = json.load(readfile)
  
  vrms = {}
  for d in dialogs:
    for speech in d:
      vrm = speech['vrm'][-1]
      if vrm in vrms:
        vrms[vrm] += 1
      else:
        vrms[vrm] = 1
  
  # logger.info("Intent VRM codes...")
  # for k in vrms.keys():
  #   logger.info(f"{k}: {vrms[k]}")

  X = np.array(sorted(vrms, key=vrms.get, reverse=True))
  Y = np.array(sorted(vrms.values(), reverse=True))
  total_vrms = sum(vrms.values())
  mpl.rc('font', **{'size': 12})
  plt.title("Intent VRMs in script")
  plt.bar(X, Y, 1, facecolor="#ef5350", edgecolor="white")
  for x,y in zip(X, Y):
    plt.text(x, y+0.5, '%d' % y, ha='center', va='bottom')
    if y == Y[-1]:
      plt.text(x, y+24, '(%.1f%%)' % (y / total_vrms * 100), ha='center', va='bottom', color="#999999")
    else:
      plt.text(x, y-30, '%.1f%%' % (y / total_vrms * 100), ha='center', va='bottom', color="white")

  plt.margins(0.05, 0.1)
  plt.show()
  



## Main
logger = uu.get_custom_logger('toolbox', log_file)

# create_word_list(input_file, output_file)
# create_skip_grams(words_file, sentences_file, output_file, log_per=log_per)
# get_info_of_sentences(sentences_file, 500000)
# make_omitted_sentences(sentences_file, '../results/reduce3-omitted-500000-5.txt', 500000, 5)
# convert_script_json_txt(directory, output_file, log_file)
# count_sentences_in_jsons(directory, log_file)
# split_sentences_in_txt(input_file, output_file, log_file)
# get_word_frequency_json(input_file, output_file, 50000)
# show_counts_log_scale(output_file)
# select_with_word_txt(output_file, split_output_file, '네', logger)
# remove_less_frequent_words(input_file, words_file, output_file, logger)
# adjust_sentence_len(input_file, output_file)
# extract_script_from_vrm(vrm_directory)
# vrm_script_to_json(input_file, output_file, logger)
# remove_duplicates_vrm_json(input_file, output_file, logger)
# test_integrity_vrm_json(input_file, logger)
count_intent_vrm_json(input_file, logger)