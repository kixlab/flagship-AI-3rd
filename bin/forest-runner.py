from utils.logger import Logger
from utils.data import create_vrm_dataset
from utils.word2vec import load_word2vec_wv
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.externals import joblib

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
forest = RandomForestClassifier(n_estimators=100)

# Create Training dataset
data = create_vrm_dataset(input_fn, wv, logger)

# Train Random Forest model
forest = forest.fit(data['X_train_t'].reshape(-1, 3900), data['Y_train_t'].reshape(-1))
joblib.dump(forest, 'test.pkl')

# joblib model load
# clf = joblib.load('filename.pkl')