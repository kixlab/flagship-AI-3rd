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
input_file = '../dataset/vrm/vrm_data_fixed_final.json'
output_name = '../dataset/vrm/vrm_single'
# output_file = '../results/vrm_data_fixed_final.json'
log_file = '../logs/toolbox.log'

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


## Main
logger = uu.get_custom_logger('toolbox', log_file)

# extract_script_from_vrm(vrm_directory)
# vrm_script_to_json(input_file, output_file, logger)
# remove_duplicates_vrm_json(input_file, output_file, logger)
# test_integrity_vrm_json(input_file, logger)
# count_intent_vrm_json(input_file, logger)
convert_vrms_to_text(input_file, output_name, logger)
