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
tester_name = "180914-axes-lstm-128"

models = {
  'disclosure': "../models/180907-lstm-128-disclosure-15-0.78.hdf5",
  'source': "../models/180911-lstm-128-source-15-0.75.hdf5",
  'frame': "../models/180914-lstm-128-frame-15-0.65.hdf5",
  'presumption': "../models/180914-lstm-128-presumption-10-0.76.hdf5"
}
embedding_fn = '../results/GoogleNews-vectors-negative300.bin'
data_fn = '../dataset/vrm/vrm-single-tokenized-v3.json'

output_fn = f'../results/{tester_name}.csv'
output_stats = f'../results/{tester_name}_stat.csv'

# Initiate Logger
logger = Logger(tester_name)

# Load word2vec
logger.console("Load word2vec embedding...")
wv = load_word2vec_wv(embedding_fn)

# Append data for output
stats_data = {
  'index': ['accuracy', 'f1', 'recall', 'precision']
}

for m in models:
  if m == 'disclosure':
    data = create_vrm_dataset(data_fn, wv, logger, get_raw_test=True)
  elif m == 'source' or m == 'frame' or m == 'presumption':
    data = create_vrm_dataset(data_fn, wv, logger, get_raw_test=True, target_preset=m)

  model = load_model(models[m])
  X_test_t = data['X_test_t']
  Y_test_t = data['Y_test_t']
  val_predict = (np.asarray(model.predict(X_test_t))).round()
  val_targ = Y_test_t
  val_f1 = f1_score(val_targ, val_predict)
  val_recall = recall_score(val_targ, val_predict)
  val_precision = precision_score(val_targ, val_predict)
  val_acc = accuracy_score(val_targ, val_predict)

  stats_data[m] = [val_acc, val_f1, val_recall, val_precision]
  

# Write csv file with print_data
for k in stats_data:
  print(f"{k}: {len(stats_data[k])}")
df = pd.DataFrame(stats_data)
df.to_csv(output_stats, sep=',')
