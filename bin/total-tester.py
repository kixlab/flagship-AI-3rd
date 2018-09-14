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
tester_name = "180914-total-runner-disclosure"

models = {
  'lstm': "../models/180907-lstm-128-disclosure-15-0.78.hdf5",
  'forest': '../models/180907-forest-1000-disclosure.pkl',
  'svm': '../models/180910-svm-basic-disclosure.pkl'
}
embedding_fn = '../results/GoogleNews-vectors-negative300.bin'
data_fn = '../dataset/vrm/vrm-single-tokenized-v3.json'

output_fn = f'../results/{tester_name}.csv'
output_statstics = f'../results/{tester_name}_stat.csv'

# Initiate Logger
logger = Logger(tester_name)

# Load word2vec
logger.console("Load word2vec embedding...")
wv = load_word2vec_wv(embedding_fn)

# Create Training dataset
data = create_vrm_dataset(data_fn, wv, logger, get_raw_test=True)

# Append data for output
print_data = {
  'X_test_raw': data['X_test_raw'],
  'tag_test': data['tag_test'],
  'Y_test': data['Y_test']
}
stats_data = {}

# Test with trained models
for k in models:
  if k == 'lstm':
    model = load_model(models[k])
    X_test_t = data['X_test_t']
    Y_test_t = data['Y_test_t']
    val_predict = model.predict(X_test_t)
    val_targ = Y_test_t
    print_data[k] = model.predict(X_test_t).reshape(-1)
  else:
    model = joblib.load(models[k])
    X_test_t = flatten_once(data['X_test_t'])
    Y_test_t = flatten_once(data['Y_test_t'])
    val_predict = model.predict(X_test_t)
    val_targ = Y_test_t
    print_data[k] = val_predict
  

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
