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
from gensim.models import KeyedVectors
from random import shuffle


## Varaibles
# directory = '../results/script_filtered.txt'
vrm_directory = '../scripts/VRMtraining'
# input_file = '../dataset/vrm/vrm_single_content.txt'
input_file = '../dataset/vrm/vrm_single_tokenized_v3.txt'
# output_name = '../dataset/vrm/vrm_test.txt'
model_path = '../results/GoogleNews-vectors-negative300.bin'
output_file = "../dataset/vrm/vrm-single-tokenized-v3.json"
log_file = '../logs/toolbox.log'
content_file = '../dataset/vrm/vrm_single_tokenized_v3.txt'
vrm_file = '../dataset/vrm/vrm_single_vrm.txt'

## Functions

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
  logger.info(
      f"{len(contents) - len(results)} of {len(contents)} are removed!")


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
      s = re.sub('([0-9]+[:][0-9]+)',
                 (lambda obj: obj.string.replace(':', ' ')), s)

      # Split content and VRM tag
      content, vrm = s[:-2].strip(), s[-2:]

      # If there is info for speaker, normalize it like 'A', 'B', ...
      if ':' in content:
        raw_speaker, content = content.split(
            ':')[0].strip(), content.split(':')[1].strip()
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
  logger.info(
      f"Remove duplicates in vrm.json: {len(possible_dup)} > {len(removed_dup)}")

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
  logger.info(
      f'Invalid speaker: {invalid_speaker_num}, Invalid VRM: {invalid_vrm_num}')


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
  for x, y in zip(X, Y):
    plt.text(x, y+0.5, '%d' % y, ha='center', va='bottom')
    if y == Y[-1]:
      plt.text(x, y+24, '(%.1f%%)' % (y / total_vrms * 100),
               ha='center', va='bottom', color="#999999")
    else:
      plt.text(x, y-30, '%.1f%%' % (y / total_vrms * 100),
               ha='center', va='bottom', color="white")

  plt.margins(0.05, 0.1)
  plt.show()


def convert_vrms_to_text(input_fn, output_name, logger):
  with open(input_fn, 'r') as readfile:
    dialogs = json.load(readfile)

  content = []
  vrms = []
  for speeches in dialogs:
    for s in speeches:
      content.append(s['utterance'])
      vrms.append(s['vrm'])

  with open(output_name + '_content.txt', 'w') as writefile:
    writefile.write(os.linesep.join(content))
  with open(output_name + '_vrm.txt', 'w') as writefile:
    writefile.write(os.linesep.join(vrms))

def tokenize_vrm_content(input_fn, output_fn, model_path ,logger):
  token_re = re.compile("[a-zA-Z]+[']*[a-zA-z]*|[0-9]")

  logger.info("Load word2vec model...")
  model = KeyedVectors.load_word2vec_format(model_path, binary="True")
  word_vectors = model.wv

  sentences = uu.load_text_file(input_fn)
  result = []
  for s in tqdm(sentences):
    tokens = token_re.findall(s)
    for t in tokens:
      try:
        word_vectors.get_vector(t)
      except KeyError as e:
        logger.info(f'"{t}" is removed.')
        tokens.remove(t)
    result.append(' '.join(tokens))
  
  with open(output_fn, 'w') as writefile:
    writefile.write(os.linesep.join(result))

def _get_proportion_indexes(arr, percents):
  total_arr = sum(arr)
  sum_arr = 0
  results = [None for i in range(len(percents))]
  for idx, v in enumerate(arr):
    sum_arr += v
    for i_idx, p in enumerate(percents):
      if results[i_idx] == None and sum_arr >= total_arr * p:
        results[i_idx] = idx
  return results

def draw_word_frequency_plot(input_fn, logger):
  sentences = uu.load_text_file(input_fn, as_words=True)

  count = {}
  for s in sentences:
    length = len(s)
    if length == 0:
      continue
    if length in count:
      count[length] += 1
    else:
      count[length] = 1
  
  logger.info('Drawing plot...')
  count_list = sorted(count.items())
  x, y = zip(*count_list)
  # i_25, i_50, i_75 = get_three_points(y)
  [i_25, i_50, i_75, i_90, i_95, i_99] = _get_proportion_indexes(
      y, [.25, .50, .75, .90, .95, .99])

  plt.plot(x, y, alpha=0.5)
  plt.scatter(x, y, s=10)
  plt.title(f'#. words in tokenized VRM script sentences')
  plt.xlabel("#. of words")
  plt.ylabel("Counts")
  plt.annotate(f"25% Value: {x[i_25]}",
              xy=(x[i_25], y[i_25]), xytext=(40, 30), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.annotate(f"50% Value: {x[i_50]}",
              xy=(x[i_50], y[i_50]), xytext=(40, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.annotate(f"75% Value: {x[i_75]}",
              xy=(x[i_75], y[i_75]), xytext=(40, 30), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.annotate(f"90% Value: {x[i_90]}",
              xy=(x[i_90], y[i_90]), xytext=(40, 50), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.annotate(f"95% Value: {x[i_95]}",
              xy=(x[i_95], y[i_95]), xytext=(40, 35), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.annotate(f"99% Value: {x[i_99]}",
              xy=(x[i_99], y[i_99]), xytext=(30, 20), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.annotate(f"End Value: {x[-1]}",
              xy=(x[-1], y[-1]), xytext=(-60, 70), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.show()

def generate_usable_vrm_set(content_fn, vrm_fn, output_fn, logger):
  contents = uu.load_text_file(content_fn)
  vrms = uu.load_text_file(vrm_fn)
  vrm_sets = []
  for c, vrm in zip(contents, vrms):
    if contents != '':
      vrm_sets.append({
        'content': c,
        'vrm': vrm
      })
  shuffle(contents)
  with open(output_fn, 'w') as writefile:
    json.dump(vrm_sets, writefile)


## Main
logger = uu.get_custom_logger('toolbox', log_file)

# extract_script_from_vrm(vrm_directory)
# vrm_script_to_json(input_file, output_file, logger)
# remove_duplicates_vrm_json(input_file, output_file, logger)
# test_integrity_vrm_json(input_file, logger)
# count_intent_vrm_json(input_file, logger)
# convert_vrms_to_text(input_file, output_name, logger)
# tokenize_vrm_content(input_file, output_file, model_path, logger)
# draw_word_frequency_plot(input_file, logger)
generate_usable_vrm_set(content_file, vrm_file, output_file, logger)