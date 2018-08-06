# Gensim Tester v0.1.1

from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import unicodedata
import _uu as uu
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import json
import operator
from sklearn.manifold import TSNE

## Variables
file_name = '180731-padded-500000sent_final'
dict_fn = os.path.join(uu.get_base_path(), 'results', file_name + '.json')
file_path = os.path.join(uu.get_base_path(), 'results', file_name)
log_path = os.path.join(uu.get_base_path(), 'logs', file_name)

## Functions
def preformat_cjk (string, width, align='<', fill=' '):
    count = (width - sum(1 + (unicodedata.east_asian_width(c) in "WF")
                         for c in string))
    return {
        '>': lambda s: fill * count + s,
        '<': lambda s: s + fill * count,
        '^': lambda s: fill * (count / 2)
                       + s
                       + fill * (count / 2 + count % 2)
}[align](string)

def draw_plot(model, filename):
  try:
    mpl.rc('font', family='SpoqaHanSans')
    with open(filename, encoding="utf-8") as json_file:
      word_dict = json.loads(json_file.read())
    plot_words = sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)[:200]
    vocab = list(map(lambda x:x[0], plot_words))
    X = model.wv[vocab]
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)
    X_tsne = tsne.fit_transform(X)
    df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

    fig = plt.figure(figsize=(15, 15), dpi=72)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df['x'], df['y'])
    for word, pos in df.iterrows():
      ax.annotate(word, pos)

    plt.title(filename)
    # plt.show()
    plt.savefig('test.png')
  except ImportError:
    print("Please install sklearn and matplotlib to visualize embeddings.")

def start_routine_get_similar(model, logger):
  word_vectors = model.wv

  logger.info('=' * 30)
  input_str = input('Similar 5 words with ')
  while(input_str != 'exit'):
    logger.debug(f'Similar 5 words with {input_str}')
    logger.info('-' * 30)
    logger.info('%s | %s' % (preformat_cjk('Word', 10), 'cosine_similarity'))
    logger.info('-' * 30)
    for word in model.most_similar(positive=[input_str], topn=5):
      logger.info('%s | %f' % (preformat_cjk(word[0], 10), word[1]))
    logger.info('=' * 30)
    input_str = input('Similar 5 words with ')

logger = uu.get_custom_logger(file_name, log_path)
model = Word2Vec.load(file_path)
print(f'Load file: {file_path}')

# start_routine_get_similar(model, logger)
draw_plot(model, dict_fn)
