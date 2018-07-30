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
INPUT_FN = os.path.join(BASE_PATH, 'results/test.txt')
PLOT_FN = os.path.join(BASE_PATH, 'screenshots/words-test.png')
OUTPUT_FN = os.path.join(BASE_PATH, 'results/test_result.txt')
LOG_FN = os.path.join(BASE_PATH, 'logs/preprocessor.log')
MAX_SENTENCE_LENGTH = 0  # If 0, the maximum is max(len(sentences))

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

  max_length = MAX_SENTENCE_LENGTH if MAX_SENTENCE_LENGTH > 0 else max(count.keys())
  logger.info(f"Max padding: {max_length}")

  with open(OUTPUT_FN, 'w') as writefile:
    for s in tqdm(ss):
      length = len(s)
      if length == 0:
        writefile.write(os.linesep)
      else:
        padded_s = ['0'] * (max_length - length) + s
        writefile.write(' '.join(padded_s) + os.linesep)

## Main
logger = uu.get_custom_logger('preprocessor', LOG_FN)

# Load sentences
logger.info('Load files...')
sentences = uu.load_text_file(INPUT_FN, ignore_first=True)
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
logger.info('Drawing plot...')
count_list = sorted(count.items())
x, y = zip(*count_list)

plt.plot(x, y)
plt.title('Before koNLPy')
plt.xlabel("#. of words")
plt.ylabel("Counts")
plt.annotate(f"End Value: {x[-1]}",
  xy=(x[-1], y[-1]), xytext=(-90, 50), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
plt.savefig(PLOT_FN)

# Padding with 0 & shortening too long sentences
pad_0_and_save(tk_sentences)

logger.info('All processes are finished!')