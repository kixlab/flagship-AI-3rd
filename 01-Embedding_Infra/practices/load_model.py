import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from sklearn.manifold import TSNE
from scipy import spatial

## Variables
plot_only = 50
embedding_size = 10

IS_COLAB = False

if (not IS_COLAB):
	# 파일 경로 설정
	dirname = os.path.dirname(__file__)

    # 이 파일과 같은 경로에 한글 폰트 파일을 넣어주세요
	font_filename = os.path.join(dirname, 'SpoqaHanSansRegular.ttf') # 폰트 이름 바꾸기

    # matplot 에서 한글을 표시하기 위한 설정
	font_name = matplotlib.font_manager.FontProperties(
        fname=font_filename
                ).get_name()
	font_filename = os.path.join(dirname, 'SpoqaHanSansRegular.ttf') # 폰트 이름 바꾸기

    # matplot 에서 한글을 표시하기 위한 설정
	font_name = matplotlib.font_manager.FontProperties(
        fname=font_filename
                ).get_name()
	matplotlib.rc('font', family=font_name)

# Load word_dict from csv file
# word_dict = {}
# with open('sci-total-dict-plain.csv') as csvfile:
#   reader = csv.reader(csvfile)
#   for row in reader:
#     word_dict[row[0]] = int(row[1])
# word_dict_reverse = dict(zip(word_dict.values(), word_dict.keys()))
# voc_size = len(word_dict)

# Load saved model meta-data
# dirname = os.path.dirname(__file__)

tf.reset_default_graph()
imported_meta = tf.train.import_meta_graph("model/test_final.meta")

with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init)

  imported_meta.restore(sess, tf.train.latest_checkpoint('model/'))
  word_dict_array = list(map(lambda x: x.decode('utf-8'), sess.run('word_dict_array:0')))
  trained_embeddings = sess.run('embeddings:0')

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

def get_nearby_elements(embeddings, index, num):
  e_size = len(embeddings)
  tree = spatial.KDTree(embeddings)
  find_array = [0] * e_size
  find_array[index] 

try:
  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)
  plot_only = len(word_dict_array) if (len(word_dict_array) < plot_only) else plot_only
  low_dim_embs = tsne.fit_transform(trained_embeddings[:plot_only, :])
  # labels = [word_dict_reverse[i] for i in range(plot_only)]
  labels = word_dict_array[:plot_only]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print("Please install sklearn and matplotlib to visualize embeddings.")
