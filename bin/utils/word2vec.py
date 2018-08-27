from gensim.models import KeyedVectors

def load_word2vec_wv(model_fn, binary=True):
  embedding_model = KeyedVectors.load_word2vec_format(model_fn, binary="True")
  word_vector = embedding_model.wv
  return word_vector