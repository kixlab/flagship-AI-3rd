from utils.logger import Logger
from utils.data import create_vrm_dataset, flatten_once
from utils.word2vec import load_word2vec_wv
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from keras.models import load_model

# Variables
logger_name = "180900-total-runner"

models = {
  'lstm': "../models/180907-lstm-128-disclosure-15-0.78.hdf5",
  'forest': '../models/forest-1000.pkl',
  # 'svm': ''
}
embedding_fn = '../results/GoogleNews-vectors-negative300.bin'
data_fn = '../dataset/vrm/vrm-single-tokenized-v3.json'

output_fn = '../results/180907-disclosure-test.csv'

# Initiate Logger
logger = Logger(logger_name)

# Load word2vec
logger.console("Load word2vec embedding...")
wv = load_word2vec_wv(embedding_fn)

# Create Training dataset
data = create_vrm_dataset(data_fn, wv, logger, get_raw_test=True)

# Append data for output
print_data = {
  'X_test_raw': data['X_test_raw'],
  'Y_test': data['Y_test']
}

# Test with trained models
for k in models:
  if k == 'lstm':
    model = load_model(models[k])
    print_data['lstm'] = model.predict(data['X_test_t']).reshape(-1)
    
  elif k == 'forest':
    model = joblib.load(models[k])
    X_test_t = flatten_once(data['X_test_t'])
    Y_test_t = flatten_once(data['Y_test_t'])
    val_predict = model.predict(X_test_t)
    val_targ = Y_test_t
    print_data['forest'] = val_predict

# Write csv file with print_data
for k in print_data:
  print(f"{k}: {len(print_data[k])}")
df = pd.DataFrame(print_data)
df.to_csv(output_fn, sep=',')


# val_predict = (np.asarray(model.predict(X_test_t))).round()
# val_targ = Y_test_t
# val_f1 = f1_score(val_targ, val_predict)
# val_recall = recall_score(val_targ, val_predict)
# val_precision = precision_score(val_targ, val_predict)
# val_acc = accuracy_score(val_targ, val_predict)
# logger.write(f"acc: {val_acc}, f1: {val_f1}, recall: {val_recall}, precision: {val_precision}")
