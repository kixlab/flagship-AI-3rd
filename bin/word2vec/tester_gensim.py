# Gensim Tester v0.1

from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

model = Word2Vec.load('../../results/180712-gensim-300000sent_final')

input_str = input('단어를 입력해주세요: ')
while(input_str != 'exit'):
  print(model.most_similar(positive=[input_str], topn=10))
  input_str = input()
