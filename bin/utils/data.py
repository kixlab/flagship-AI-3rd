import re
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



def create_vrm_dataset(input_fn, wv, logger, max_len=13, train_ratio=0.85):
  logger.write(f"Load vrm data: {input_fn}...")
  data = utils.file.read_json(input_fn)

  # Divide data into positive or negative ones.
  X_positive = []
  X_negative = []
  for d in data:
    content = pad_to_tokens(d['content'].split(' '), max_len)
    if d['vrm'].endswith('D'):
      X_positive.append(content)
    else:
      X_negative.append(content)

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

  print(f"Data: X_train - {X_train_t.shape}, X_test - {X_test_t.shape}")
  print(f"Data: Y_train - {Y_train_t.shape}, Y_test - {Y_test_t.shape}")

  # return X_train_t, X_test_t, Y_train_t, Y_test_t
  return {
    'X_train_t': X_train_t,
    'X_test_t': X_test_t,
    'Y_train_t': Y_train_t,
    'Y_test_t': Y_test_t
  }

def _get_embedding_vector(word, wv):
  try:
    return wv[word]
  except KeyError:
    return [0] * 300
