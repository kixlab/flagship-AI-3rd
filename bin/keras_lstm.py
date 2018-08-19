# Refer http://3months.tistory.com/168

# Simple LSTM model using keras
import os
import universal_utils as uu
import numpy as np
import matplotlib.pyplot as plt
import json
from gensim.models import Word2Vec
from keras.layers import LSTM, Dropout
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping, Callback, TensorBoard, ModelCheckpoint
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from gensim.models import KeyedVectors

## Variables
embedding_fn = 'GoogleNews-vectors-negative300.bin'
input_fn = 'vrm/vrm-single-tokenized-v3.json'
positive_fn = 'ne_positive.txt'
negative_fn = 'ne_negative.txt'
ye_fn = 'ye_positive.txt'
logs_fn = 'keras_lstm-test.log'
# save_fn = 'disclosure_128'
train_ratio = 0.85
max_len = 13

base_path = uu.get_base_path()

## Paths
EMBEDDING_PATH = os.path.join(base_path, 'results', embedding_fn)
POSITIVE_PATH = os.path.join(base_path, 'results/lstm-testset', positive_fn)
NEGATIVE_PATH = os.path.join(base_path, 'results/lstm-testset', negative_fn)
# SPECIAL_PATH = os.path.join(base_path, 'results/lstm-testset', ye_fn)
LOG_PATH = os.path.join(base_path, 'logs', logs_fn)
INPUT_PATH = os.path.join(base_path, 'dataset', input_fn)
# SAVE_PATH = os.path.join(base_path, 'results', save_fn)

## Classes & Functions
class evaluateIter(Callback):
  def on_epoch_end(self, epoch, logs={}):
    scores = self.model.evaluate(X_test_t, Y_test_t, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

class Metrics(Callback):
  def on_train_begin(self, logs={}):
    self.val_f1s = []
    self.val_recalls = []
    self.val_precisions = []
    
  def on_epoch_end(self, epoch, logs={}):
    val_predict = (np.asarray(self.model.predict(X_test_t))).round()
    val_targ = Y_test_t
    _val_f1 = f1_score(val_targ, val_predict)
    _val_recall = recall_score(val_targ, val_predict)
    _val_precision = precision_score(val_targ, val_predict)
    self.val_f1s.append(_val_f1)
    self.val_recalls.append(_val_recall)
    self.val_precisions.append(_val_precision)
    print(" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
    return

def _pad_sentence_into_words(sentence, max_len):
  tokens = sentence.split(' ')
  if len(tokens) < max_len:
    result = [' '] * (max_len - len(tokens)) + tokens
  else:
    result = tokens[:max_len]
  return result

def _get_embedding_vector(word, wv):
  try:
    return wv[word]
  except KeyError as e:
    return [0] * 300

def create_basic_dataset():
  # Load train & test data
  positive_set = uu.load_text_file(POSITIVE_PATH, as_words=True)
  negative_set = uu.load_text_file(NEGATIVE_PATH, as_words=True)

  X_train = positive_set[:int(len(positive_set) * train_ratio)] + \
            negative_set[:int(len(negative_set) * train_ratio)]
  X_test = positive_set[int(len(positive_set) * train_ratio):] + \
          negative_set[int(len(negative_set) * train_ratio):]
  Y_train = [[1]] * int(len(positive_set) * train_ratio) + \
            [[0]] * int(len(negative_set) * train_ratio)
  Y_test = [[1]] * (len(positive_set) - int(len(positive_set) * train_ratio)) + \
          [[0]] * (len(negative_set) - int(len(negative_set) * train_ratio))

  # print(f'{len(X_train)}, {len(X_test)}, {len(Y_train)}, {len(Y_test)}')
  # print(X_test[916])

  # Translate words into vectors
  embedding_model = Word2Vec.load(EMBEDDING_PATH).wv
  X_train_t = np.array(list(map(lambda x: embedding_model[x], X_train)))
  X_test_t = np.array(list(map(lambda x: embedding_model[x], X_test)))
  Y_train_t = np.array(Y_train)
  Y_test_t = np.array(Y_test)

  # print(f"Train set: X - {X_train_t.shape}, Y - {Y_train_t.shape}")
  # print(f"Test set: X - {X_test_t.shape}, Y - {Y_test_t.shape}")

  return X_train_t, X_test_t, Y_train_t, Y_test_t

def create_vrm_dataset(input_fn, model_fn, logger):
  logger.info(f"Load vrm data: {input_fn}...")
  with open(input_fn, 'r') as readfile:
    data = json.load(readfile)

  logger.info(f"Load Word2Vec model: {model_fn}...")
  embedding_model = KeyedVectors.load_word2vec_format(model_fn, binary="True")
  word_vector = embedding_model.wv
  
  # Divide data into positive or negative ones.
  X_positive = []
  X_negative = []
  for d in data:
    content = _pad_sentence_into_words(d['content'], max_len)
    if d['vrm'].endswith('D'):
      X_positive.append(content)
    else:
      X_negative.append(content)

  logger.info(f"Positive set: {len(X_positive)}")
  logger.info(f"Negative set: {len(X_negative)}")

  # Turn data into vectors
  # X_positive_t = np.array(list(map(_get_embedding_vector(vw=embedding_model), X_positive)))
  # X_negative_t = np.array(list(map(_get_embedding_vector(vw=embedding_model), X_negative)))
  X_positive_t = []
  X_negative_t = []
  for words in X_positive:
    X_positive_t.append([_get_embedding_vector(w, word_vector) for w in words])
  for words in X_negative:
    X_negative_t.append([_get_embedding_vector(w, word_vector) for w in words])

  # print(f"Data: positive - {X_positive_t.shape}, negative - {X_negative_t.shape}")

  # Divide into train & test set
  positive_train_len = int(len(X_positive_t) * train_ratio)
  negative_train_len = int(len(X_negative_t) * train_ratio)
  X_train_t = np.array(X_positive_t[:positive_train_len] + X_negative_t[:negative_train_len])
  X_test_t = np.array(X_positive_t[positive_train_len:] + X_negative_t[negative_train_len:])
  Y_train_t = np.array([[1]] * positive_train_len + [[0]] * negative_train_len)
  Y_test_t = np.array([[1]] * (len(X_positive) - positive_train_len) + [[0]] * (len(X_negative) - negative_train_len))

  print(f"Data: X_train - {X_train_t.shape}, X_test - {X_test_t.shape}")
  print(f"Data: Y_train - {Y_train_t.shape}, Y_test - {Y_test_t.shape}")

  return X_train_t, X_test_t, Y_train_t, Y_test_t


## Main
logger = uu.get_custom_logger('keras_lstm', LOG_PATH)

X_train_t, X_test_t, Y_train_t, Y_test_t = create_vrm_dataset(INPUT_PATH, EMBEDDING_PATH, logger)

# Create LSTM Model
K.clear_session()
model = Sequential()
model.add(LSTM(128, input_shape=(13, 300)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Model Fitting
eval_iter = evaluateIter()
metrics = Metrics()
# tbCallBack = TensorBoard(
#     log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
filepath = "disclosure-128-{epoch:02d}-{val_acc:.2f}.hdf5"
modelcheckpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=0,
                save_best_only=False, save_weights_only=False, mode='auto', period=1)
early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)
callback_list = [early_stop, eval_iter, metrics, modelcheckpoint]
model.fit(X_train_t, Y_train_t, validation_data=(X_test_t, Y_test_t), epochs=10, batch_size=64, verbose=1, callbacks=callback_list)

# scores = model.evaluate(X_test_t, Y_test_t, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))
