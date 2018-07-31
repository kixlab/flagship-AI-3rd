## Pre-processor for training LSTM model

import os
import re
import _uu as uu
from tqdm import tqdm
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pylab as plt

## Global
BASE_PATH = uu.get_base_path()

## Variables
INPUT_FN = os.path.join(BASE_PATH, 'results/script_filtered_splited.txt')
PLOT_FN = os.path.join(BASE_PATH, 'screenshots/180731-500000-after.png')
OUTPUT_FN = os.path.join(BASE_PATH, 'results/180731-padded-500000-splited.txt')
LOG_FN = os.path.join(BASE_PATH, 'logs/preprocessor.log')
SENTENCES_LENGTH = 500000
MAX_WORDS_LENGTH = 16  # If 0, the maximum is max(len(sentences))

## Fucntions
def reduce_word(w, num=3):
  if (len(w) > num):
    return w[:num] + '-'
  else:
    return w

def tokenize_sentence(s):
  # regex rules
  TOKEN_RE = re.compile('([ㄱ-ㅎㅏ-ㅢ가-힣]+|\.\.\.|\.|\?|\!|\,)')
  IN_BRACKETS_RE = re.compile('\(.*?\)')  # Limitation on nested brackets like '(a(b)c)'
  ELLIPSIS_RE = re.compile('\.\.+|…')
  NUMBER_RE = re.compile('[0-9]+[.][0-9]+')

  # Remove words in brackets
  result = IN_BRACKETS_RE.sub(' ', s)

  # make all ellipses words into single word
  result = ELLIPSIS_RE.sub('...', s)

  # Replace numbers to single Hangul form
  reuslt = NUMBER_RE.sub('숫자', s)

  # Divide sentence into words with regex
  result = TOKEN_RE.findall(result)

  # reduce to 3-length words
  result = list(map(reduce_word, result))

  return result

def pad_0_and_save(ss):
  logger.info('Writing results...')

  max_length = MAX_WORDS_LENGTH if MAX_WORDS_LENGTH > 0 else max(count.keys())
  logger.info(f"Max padding: {max_length}")

  with open(OUTPUT_FN, 'w') as writefile:
    for s in tqdm(ss):
      length = len(s)
      if length == 0:
        continue
      else:
        padded_s = ['0'] * (max_length - length) + s
        writefile.write(' '.join(padded_s) + os.linesep)

def get_three_points(arr):
  total_arr = sum(arr)
  sum_arr = 0
  i_25 = i_50 = i_75 = None
  for idx, v in enumerate(arr):
    sum_arr += v
    if i_25 == None and sum_arr >= total_arr * 0.25:
      i_25 = idx
    if i_50 == None and sum_arr >= total_arr * 0.5:
      i_50 = idx
    if i_75 == None and sum_arr >= total_arr * 0.75:
      i_75 = idx
  return i_25, i_50, i_75

def get_proportion_indexes(arr, percents):
  total_arr = sum(arr)
  sum_arr = 0
  results = [None for i in range(len(percents))]
  for idx, v in enumerate(arr):
    sum_arr += v
    for i_idx, p in enumerate(percents):
      if results[i_idx] == None and sum_arr >= total_arr * p:
        results[i_idx] = idx
  return results

def draw_plot(count):
  logger.info('Drawing plot...')
  count_list = sorted(count.items())
  x, y = zip(*count_list)
  # i_25, i_50, i_75 = get_three_points(y)
  [i_25, i_50, i_75, i_90, i_95, i_99] = get_proportion_indexes(
      y, [.25, .50, .75, .90, .95, .99])

  plt.plot(x, y, alpha=0.5)
  plt.scatter(x, y, s=10)
  plt.title(f'After spliting - {SENTENCES_LENGTH} sentences')
  plt.xlabel("#. of words")
  plt.ylabel("Counts")
  plt.annotate(f"25% Value: {x[i_25]}",
              xy=(x[i_25], y[i_25]), xytext=(40, -10), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.annotate(f"50% Value: {x[i_50]}",
              xy=(x[i_50], y[i_50]), xytext=(40, -10), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.annotate(f"75% Value: {x[i_75]}",
              xy=(x[i_75], y[i_75]), xytext=(40, 40), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.annotate(f"90% Value: {x[i_90]}",
              xy=(x[i_90], y[i_90]), xytext=(40, 50), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.annotate(f"95% Value: {x[i_95]}",
              xy=(x[i_95], y[i_95]), xytext=(40, 35), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.annotate(f"99% Value: {x[i_99]}",
              xy=(x[i_99], y[i_99]), xytext=(40, 20), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.annotate(f"End Value: {x[-1]}",
              xy=(x[-1], y[-1]), xytext=(-90, 50), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.savefig(PLOT_FN)

## Main
logger = uu.get_custom_logger('preprocessor', LOG_FN)

# Load sentences
logger.info('Load files...')
sentences = uu.load_text_file(INPUT_FN, ignore_first=True, max_num=SENTENCES_LENGTH)
logger.info(f'Total {len(sentences)} sentences were parsed.')

# Split combined sentences

# Tokenizing with processing words
logger.info('Tokenizing...')
tk_sentences = []
for s in tqdm(sentences):
  tk_sentences.append(tokenize_sentence(s))

# Count #. of words in each sentences
logger.info('Counting...')
count = {}
for s in tqdm(tk_sentences):
  length = len(s)
  if length == 0:
    continue
  elif length in count:
    count[length] += 1
  else:
    count[length] = 1

# Draw plot
# draw_plot(count)

# Padding with 0 & shortening too long sentences
pad_0_and_save(tk_sentences)

logger.info('All processes are finished!')
