import re
from nltk.stem import LancasterStemmer
import utils.file
import numpy as np
from gensim.models import KeyedVectors

def tokenize_sentence_vrm(sentence, wv):
  token_re = re.compile("[a-zA-Z]+[']*[a-zA-z]*|[0-9]")

  tokens = token_re.findall(sentence)
  remove_words = []
  for t in tokens:
    try:
      wv.get_vector(t)
    except KeyError:
      print(f'"{t}" is removed.')
      remove_words.append(t)
  
  return list(filter(lambda ele: ele not in remove_words, tokens))

def pad_to_tokens(tokens, max_len):
  if len(tokens) < max_len:
    result = [' '] * (max_len - len(tokens)) + tokens
  else:
    result = tokens[:max_len]
  return result

def create_vrm_dataset(input_fn, wv, logger, max_len=13, train_ratio=0.85, is_intent=True, target_vrm='D', get_raw_test=False, target_preset=None):
  logger.write(f"Load vrm data: {input_fn}...")
  data = utils.file.read_json(input_fn)

  if target_preset == 'source':
    target_vrm = ['C', 'A', 'D', 'E']
  elif target_preset == 'frame':
    target_vrm = ['I', 'Q', 'A', 'D']
  elif target_preset == 'presumption':
    target_vrm = ['K', 'Q', 'E', 'D']

  def is_target_vrm(x):
    if type(target_vrm) is str:
      return x.endswith(target_vrm) if is_intent else x.startswith(target_vrm)
    elif type(target_vrm) is list:
      return x[1] in target_vrm if is_intent else x[0] in target_vrm
    else:
      return None

  # Divide data into positive or negative ones.
  X_positive = []
  X_negative = []
  tag_positive = []
  tag_negative = []
  for d in data:
    content = pad_to_tokens(d['content'].split(' '), max_len)
    if is_target_vrm(d['vrm']):
      X_positive.append(content)
      tag_positive.append(d['vrm'])
    else:
      X_negative.append(content)
      tag_negative.append(d['vrm'])

  logger.write(f"Positive set: {len(X_positive)}")
  logger.write(f"Negative set: {len(X_negative)}")

  # Turn data into vectors
  # X_positive_t = np.array(list(map(_get_embedding_vector(vw=embedding_model), X_positive)))
  # X_negative_t = np.array(list(map(_get_embedding_vector(vw=embedding_model), X_negative)))
  X_positive_t = []
  X_negative_t = []
  for words in X_positive:
    X_positive_t.append([_get_embedding_vector(w, wv) for w in words])
  for words in X_negative:
    X_negative_t.append([_get_embedding_vector(w, wv) for w in words])

  # print(f"Data: positive - {X_positive_t.shape}, negative - {X_negative_t.shape}")

  # Divide into train & test set
  positive_train_len = int(len(X_positive_t) * train_ratio)
  negative_train_len = int(len(X_negative_t) * train_ratio)
  X_train_t = np.array(
      X_positive_t[:positive_train_len] + X_negative_t[:negative_train_len])
  X_test_t = np.array(
      X_positive_t[positive_train_len:] + X_negative_t[negative_train_len:])
  Y_train_t = np.array([[1]] * positive_train_len + [[0]] * negative_train_len)
  Y_test_t = np.array([[1]] * (len(X_positive) - positive_train_len) +
                      [[0]] * (len(X_negative) - negative_train_len))
  tag_test = tag_positive[positive_train_len:] + tag_negative[negative_train_len:]

  print(f"Data: X_train - {X_train_t.shape}, X_test - {X_test_t.shape}")
  print(f"Data: Y_train - {Y_train_t.shape}, Y_test - {Y_test_t.shape}")

  result = {
    'X_train_t': X_train_t,
    'X_test_t': X_test_t,
    'Y_train_t': Y_train_t,
    'Y_test_t': Y_test_t,
    'tag_test': tag_test
  }

  if get_raw_test:
    X_test_raw = X_positive[positive_train_len:] + X_negative[negative_train_len:]
    X_test_raw = list(map(lambda x: ' '.join([ele for ele in x if ele is not ' ']), X_test_raw))
    Y_test = [1] * (len(X_positive) - positive_train_len) + \
        [0] * (len(X_negative) - negative_train_len)
    
    result['X_test_raw'] = X_test_raw
    result['Y_test'] = Y_test

  return result

def _get_embedding_vector(word, wv):
  try:
    return wv[word]
  except KeyError:
    return [0] * 300

def flatten_once(nparr):
  shape = nparr.shape
  return nparr.reshape(shape[:-2] + (-1,))

def get_dict_count(d, category_name):
  result = {}
  for item in d:
    category = item[category_name]
    if category in result.keys():
      result[category] += 1
    else:
      result[category] = 1
  return result

def tokenize_sentence_swda(s, delimiter=None):
  # Remove words in [ ... ]
  s = re.sub(r"\[[^\]]*\]", "", s)
  
  # Remove words in < ... >
  s = re.sub(r"\<[^\>]*\>", "", s)

  # Remove a character after {
  s = re.sub(r"\{[A-Za-z]([^\}]*)\}", r"\1", s)

  # Only select English and ! ?
  token_re = re.compile("[a-zA-Z]+|\!|\?")
  tokens = token_re.findall(s)

  st = LancasterStemmer()
  tokens = [st.stem(w) for w in tokens]

  if delimiter:
    return delimiter.join(tokens)

  return tokens

def count_tag(data, tags):
  result = 0
  for t in tags:
    for k in data:
      if (k['tag'].startswith(t)):
        result += data[k]
  return result