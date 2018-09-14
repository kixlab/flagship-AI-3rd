from utils.logger import Logger
from utils.data import create_vrm_dataset
from utils.word2vec import load_word2vec_wv
from models.simple_keras import SimpleKeras

# Variables
model_name = "180914-lstm-128-presumption"
input_fn = '../dataset/vrm/vrm-single-tokenized-v3.json'
model_fn = '../results/GoogleNews-vectors-negative300.bin'

# Initiate Logger
logger = Logger(model_name)

# Load word2vec
logger.console("Load word2vec embedding...")
wv = load_word2vec_wv(model_fn)

# Initiate Keras model
logger.console("Initiate Simple Keras...")
simple_keras = SimpleKeras(model_name, logger)

# Create Training dataset
data = create_vrm_dataset(input_fn, wv, logger, target_vrm=['K', 'E', 'D', 'Q'])

# Train keras model
simple_keras.train(data, epoch=10, save_per=10)