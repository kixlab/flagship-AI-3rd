from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential, load_model
import keras.backend as K
from keras.callbacks import EarlyStopping, Callback, TensorBoard, ModelCheckpoint
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import numpy as np
from utils.file import write_file

class SimpleKeras:
  def __init__(self, model_name, logger):
    self.model_name = model_name

    self.logger = logger
    self.logger.console("Start Simple Keras!")

    # Create LSTM Model
    K.clear_session()
    self.model = Sequential()
    self.model.add(LSTM(128, input_shape=(13, 300)))
    self.model.add(Dropout(0.2))
    self.model.add(Dense(1, activation='sigmoid'))
    self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    self.model.summary()
  
  def train(self, data, epoch=20, save_per=20):
    model_name = self.model_name

    # Callback Classes
    class evaluateIter(Callback):
      def on_epoch_end(self, epoch, logs={}):
        scores = self.model.evaluate(X_test_t, Y_test_t, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))

    class Metrics(Callback):
      def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []
        self.val_losses = []
        self.val_accs = []
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

      def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accs.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_accs.append(logs.get('val_acc'))

        val_predict = (np.asarray(self.model.predict(X_test_t))).round()
        val_targ = Y_test_t
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: %f — val_precision: %f — val_recall %f" %
              (_val_f1, _val_precision, _val_recall))
        return
      
      def on_train_end(self, logs={}):
        data = {
          "losses": self.losses,
          "accuracies": self.accs,
          "val_losses": self.val_losses,
          "val_accuracies": self.val_accs,
          "val_f1s": self.val_f1s,
          "val_recalls": self.val_recalls,
          "val_precisions": self.val_precisions
        }
        write_file(data, f"../results/{model_name}-results.json", is_json=True)

    X_train_t, X_test_t, Y_train_t, Y_test_t = self._load_data(data)

    eval_iter = evaluateIter()
    metrics = Metrics()

    filepath = "../models/" + self.model_name + "-{epoch:02d}-{val_acc:.2f}.hdf5"
    modelcheckpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=0,
                    save_best_only=False, save_weights_only=False, mode='auto', period=save_per)
    # early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)
    callback_list = [
      # early_stop,
      eval_iter, metrics, modelcheckpoint]
    self.model.fit(X_train_t, Y_train_t, validation_data=(X_test_t, Y_test_t), epochs=epoch, batch_size=64, verbose=1, callbacks=callback_list)

  def load_model(self, filepath):
    self.model = load_model(filepath)

  def _load_data(self, data):
    X_train_t = data['X_train_t']
    X_test_t = data['X_test_t']
    Y_train_t = data['Y_train_t']
    Y_test_t = data['Y_test_t']
    return X_train_t, X_test_t, Y_train_t, Y_test_t
