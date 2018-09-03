from utils.logger import Logger
from utils.data import create_vrm_dataset
from utils.word2vec import load_word2vec_wv
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

# Variables
model_name = "180901-lstm-64-disclosure"
input_fn = '../dataset/vrm/vrm-single-tokenized-v3.json'
model_fn = '../results/GoogleNews-vectors-negative300.bin'

# Initiate Logger
logger = Logger(model_name)

# Load word2vec
logger.console("Load word2vec embedding...")
wv = load_word2vec_wv(model_fn)

# Initiate random forest
logger.console("Initiate Random Forest")
forest = RandomForestClassifier(n_estimators=1000)

# Create Training dataset
data = create_vrm_dataset(input_fn, wv, logger)

# Train Random Forest model
# forest = forest.fit(data['X_train_t'].reshape(-1, 3900), data['Y_train_t'].reshape(-1))
# joblib.dump(forest, 'forest-1000.pkl')

# joblib model load
model = joblib.load('forest-1000.pkl')
model = forest
X_test_t = data['X_test_t'].reshape(-1, 3900)
Y_test_t = data['Y_test_t'].reshape(-1)

val_predict = (np.asarray(model.predict(X_test_t))).round()
val_targ = Y_test_t
val_f1 = f1_score(val_targ, val_predict)
val_recall = recall_score(val_targ, val_predict)
val_precision = precision_score(val_targ, val_predict)
val_acc = accuracy_score(val_targ, val_predict)
logger.write(f"acc: {val_acc}, f1: {val_f1}, recall: {val_recall}, precision: {val_precision}")