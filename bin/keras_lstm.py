# Refer http://3months.tistory.com/168

# Simple LSTM model using keras
import os
import universal_utils as uu
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from keras.layers import LSTM, Dropout
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping, Callback, TensorBoard
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

## Variables
embedding_fn = '180731-padded-500000sent_final'
positive_fn = 'ne_positive.txt'
negative_fn = 'ne_negative.txt'
ye_fn = 'ye_positive.txt'
logs_fn = 'keras_lstm-test.log'
train_ratio = 0.8

base_path = uu.get_base_path()

## Paths
EMBEDDING_PATH = os.path.join(base_path, 'results', embedding_fn)
POSITIVE_PATH = os.path.join(base_path, 'results/lstm-testset', positive_fn)
NEGATIVE_PATH = os.path.join(base_path, 'results/lstm-testset', negative_fn)
SPECIAL_PATH = os.path.join(base_path, 'results/lstm-testset', ye_fn)
LOG_PATH = os.path.join(base_path, 'logs', logs_fn)

## Classes & Functions
class evaluateIter(Callback):
  def on_epoch_end(self, epoch, logs={}):
    scores = self.model.evaluate(X_special_t, Y_special_t, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

class Metrics(Callback):
  def on_train_begin(self, logs={}):
    self.val_f1s = []
    self.val_recalls = []
    self.val_precisions = []
    
  def on_epoch_end(self, epoch, logs={}):
    val_predict = (np.asarray(self.model.predict(X_special_t))).round()
    val_targ = Y_special_t
    _val_f1 = f1_score(val_targ, val_predict)
    _val_recall = recall_score(val_targ, val_predict)
    _val_precision = precision_score(val_targ, val_predict)
    self.val_f1s.append(_val_f1)
    self.val_recalls.append(_val_recall)
    self.val_precisions.append(_val_precision)
    print(" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
    return


## Main
logger = uu.get_custom_logger('keras_lstm', LOG_PATH)

# Load train & test data
positive_set = uu.load_text_file(POSITIVE_PATH, as_words=True)
negative_set = uu.load_text_file(NEGATIVE_PATH, as_words=True)
special_set = uu.load_text_file(SPECIAL_PATH, as_words=True)

X_train = positive_set[:int(len(positive_set) * train_ratio)] + \
          negative_set[:int(len(negative_set) * train_ratio)]
X_test = positive_set[int(len(positive_set) * train_ratio):] + \
         negative_set[int(len(negative_set) * train_ratio):]
Y_train = [[1]] * int(len(positive_set) * train_ratio) + \
          [[0]] * int(len(negative_set) * train_ratio)
Y_test = [[1]] * (len(positive_set) - int(len(positive_set) * train_ratio)) + \
         [[0]] * (len(negative_set) - int(len(negative_set) * train_ratio))
X_special = special_set
Y_special = [[1]] * len(special_set)

# print(f'{len(X_train)}, {len(X_test)}, {len(Y_train)}, {len(Y_test)}')
# print(X_test[916])

# Translate words into vectors
embedding_model = Word2Vec.load(EMBEDDING_PATH).wv
X_train_t = np.array(list(map(lambda x: embedding_model[x], X_train)))
X_test_t = np.array(list(map(lambda x: embedding_model[x], X_test)))
Y_train_t = np.array(Y_train)
Y_test_t = np.array(Y_test)
X_special_t = np.array(list(map(lambda x: embedding_model[x], X_special)))
Y_special_t = np.array(Y_special)

# print(f"Train set: X - {X_train_t.shape}, Y - {Y_train_t.shape}")
# print(f"Test set: X - {X_test_t.shape}, Y - {Y_test_t.shape}")

# Create LSTM Model
K.clear_session()
model = Sequential()
model.add(LSTM(16, input_shape=(16, 50)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Model Fitting
eval_iter = evaluateIter()
metrics = Metrics()
# tbCallBack = TensorBoard(
#     log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)
model.fit(X_train_t, Y_train_t, validation_data=(X_test_t, Y_test_t), epochs=10, batch_size=64, verbose=1, callbacks=[early_stop, eval_iter, metrics])

# scores = model.evaluate(X_test_t, Y_test_t, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))
