import numpy as np
import matplotlib.pylab as plt
from utils.file import load_text_file, write_file
from utils.word2vec import load_word2vec_wv
from collections import Counter, OrderedDict
from models.bigru_crf import BigruCrf
from keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score
from utils.logger import Logger
from keras_contrib.layers import CRF
from collections import Counter

embedding_fn = '../results-nosave/GoogleNews-vectors-negative300.bin'

data_X_fn = '../data-da/swda-data-pos_X.csv'
data_y_fn = '../data-da/swda-data-pos_y.csv'
data_pos_tags = '../data-da/swda-data-pos_dict.txt'
data_y_tags = '../data-da/swda-data-y_dict.txt'
model_path = '../models/181130-test-bigrucrf.h5'

sent_len = 21

my_labels = [0,1,2,3]
my_tags = ['e', 'x', 'd', 'k']

def remove_new_lines(X):
  return 

def extract_pos_words_list(X):
  return list(map(lambda x: extract_pos_words(x), X))

def extract_pos_words(X):
  X_reduced = X[:-3]
  return list(map(lambda x: x.split('/')[1], X_reduced))

def write_dict(X, write_fn):
  pos_words = []
  for line in X:
    for pos in line:
      if pos not in pos_words:
        pos_words.append(pos)
  write_file(pos_words, write_fn)

def convert_pos_to_idx(X, tags_fn, y=False):
  pos_dict = load_text_file(tags_fn)
  result = []
  if not y:
    for line in X:
      result.append(list(map(lambda x: pos_dict.index(x) + 1, line)))
  else:
    result = [pos_dict.index(x) for x in X]
  return result, (len(pos_dict) if y else len(pos_dict) + 1)

def get_count(X):
  result = {}
  for line in X:
    for x in line:
      if x not in result.keys():
        result[x] = 1
      else:
        result[x] += 1
  return result

def get_length_count(X):
  lengths = [len(x) for x in X]
  return OrderedDict(sorted(Counter(lengths).items()))

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

def pad_X(X, max_len, pad_word=0):
  result = []
  for line in X:
    if len(line) < max_len:
      result.append([pad_word] * (max_len - len(line)) + line)
    else:
      result.append(line[:max_len])
  return np.array(result)

def convert_one_hot(x, pos_len, y=False):
  if not y:
    result = []
    for line in x:
      new = np.zeros((line.shape[0], pos_len))
      new[np.arange(line.shape[0]), line] = 1
      result.append(new)
    return np.array(result)
  else:
    new = np.zeros((x.shape[0], pos_len))
    new[np.arange(x.shape[0]), x] = 1
    return np.array(list(map(lambda x: [x], new)))

def convert_one_hot_single(idx, pos_len):
  new = np.zeros(pos_len)
  new[idx] = 1
  return new

def convert_to_integer(x):
  result = []
  for line in x:
    result.append([np.where(r == 1)[0][0] for r in line][-1])
  return result

def draw_plot(x, y):
  [i_25, i_50, i_75, i_90, i_95, i_99] = _get_proportion_indexes(
      y, [.25, .50, .75, .90, .95, .99])

  plt.plot(x, y)
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

def create_custom_objects():
    instanceHolder = {"instance": None}

    class ClassWrapper(CRF):
        def __init__(self, *args, **kwargs):
            instanceHolder["instance"] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)

    def loss(*args):
        method = getattr(instanceHolder["instance"], "loss_function")
        return method(*args)

    def accuracy(*args):
        method = getattr(instanceHolder["instance"], "accuracy")
        return method(*args)
    return {"ClassWrapper": ClassWrapper, "CRF": ClassWrapper, "loss": loss, "accuracy": accuracy}

def load_keras_model(path):
    model = load_model(path, custom_objects=create_custom_objects())
    return model

def convert_sent_to_vectors(sent, word_vec, pos_dict_fn, offset=3, embedding_len=300, sent_len=21):
  pos_dict = load_text_file(pos_dict_fn)
  result = []
  pos_len = len(pos_dict) + 1
  for item in sent[:-offset]:
    word, pos = item.split('/')
    try:
      vec_word = word_vec[word]
    except KeyError:
      vec_word = np.zeros(embedding_len)
    vec = np.append(vec_word, convert_one_hot_single(pos_dict.index(pos) + 1, pos_len))
    result.append(vec)

  if (sent_len - len(result)) > 0:
    vec_pad = np.zeros((sent_len - len(result), embedding_len + pos_len))
    return np.concatenate((vec_pad, result))
  return result[:sent_len]

# Load logger
logger = Logger('crf-wv-runner')
wv = load_word2vec_wv(embedding_fn)

# Read words
X = load_text_file(data_X_fn, as_words=True)

X = [x for x in X if x[0] is not '']
# X = extract_pos_words_list(X)

X = np.array([convert_sent_to_vectors(x, wv, data_pos_tags, sent_len=sent_len) for x in X])
print(X.shape)

y = load_text_file(data_y_fn)
y = [x for x in y if len(x) > 0]

# Draw length plot
# length_count = get_length_count(X)
# draw_plot(list(length_count.keys()), list(length_count.values()))

# Write dict
# write_dict(X, data_pos_tags)
# write_dict(y, data_y_tags)

logger = Logger('bigrucrf')
pos_len = len(load_text_file(data_pos_tags)) + 1

# Convert pos array into index array
# X, pos_len = convert_pos_to_idx(X, data_pos_tags)
# X = pad_X(X, sent_len)
# X = np.array(X)
# # X = convert_one_hot(np.array(X), pos_len)
# # print(X.shape)
y, ans_len = convert_pos_to_idx(y, data_y_tags, y=True)
# print(ans_len)
y = convert_one_hot(np.array(y), ans_len, y=True)
# u = np.array(y)
# print(y[:10])

# Split data
split_index = int(X.shape[0] * .85)
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]
# print(X_train[100])
# print(X_test.shape)

# Keras
model = BigruCrf(pos_len, sent_len)
model.fit(X_train, y_train, X_test, y_test, epochs=5)
model.save(model_path)
del model

# Test
model = load_keras_model(model_path)
y_pred = model.predict(X_test)
y_test = convert_to_integer(y_test)
y_pred = convert_to_integer(y_pred)

# print(Counter(y_test))
# sum = 0
# for i in range(len(y_test)):
#   if (np.array_equal(y_pred[-1], y_test[-1])):
#     sum += 1
# print(f'acc: {sum / len(y_test)}')
logger.write('accuracy %s' % accuracy_score(y_pred, y_test))
logger.write(classification_report(y_test, y_pred, labels=my_labels, target_names=my_tags))
