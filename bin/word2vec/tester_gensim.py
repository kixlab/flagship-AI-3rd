# Gensim Tester v0.1

from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import unicodedata
import _uu as uu
import os

## Variables
file_name = '180713-gensim-500000sent_final'
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

start_routine_get_similar(model, logger)