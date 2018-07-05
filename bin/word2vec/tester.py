# Word2vec tester - v0.1

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from sklearn.manifold import TSNE
from sklearn.neighbors import BallTree
from scipy import spatial

## Variables
plot_num = 50
embedding_size = 10

IS_COLAB = False

## Functions for testing model


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


def start_getting_nearby_elements(k):
  tree = BallTree(embeddings)
  print("Tree build completed; You can search nearest %d words." % k)
  print("Type 'exit' to exit the program.")
  input_string = input()
  while(input_string != 'exit'):
    if (input_string in word_dict_array):
      input_index = word_dict_array.index(input_string)
      dist, ind = tree.query([embeddings[input_index]], k=k+1)
      print("Similiar %d words with %s" % (k, input_string))
      for i in range(1, k+1):
        print("%s: %f" % (word_dict_array[ind[0][i]], dist[0][i]))
      # print(ind)
      # print(dist)
    else:
      print("Input word is not in word_dict!")
    input_string = input("Please the word to test: ")


def draw_plot(plot_num):
  try:
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)
    plot_only = len(word_dict_array) if (len(word_dict_array) < plot_num) else plot_num
    low_dim_embs = tsne.fit_transform(embeddings[:plot_only, :])
    # labels = [word_dict_reverse[i] for i in range(plot_only)]
    labels = word_dict_array[:plot_only]
    plot_with_labels(low_dim_embs, labels)

  except ImportError:
    print("Please install sklearn and matplotlib to visualize embeddings.")


if (not IS_COLAB):
	# 파일 경로 설정
	dirname = os.path.dirname(__file__)

    # 이 파일과 같은 경로에 한글 폰트 파일을 넣어주세요
	font_filename = os.path.join(dirname, 'SpoqaHanSansRegular.ttf')  # 폰트 이름 바꾸기

    # matplot 에서 한글을 표시하기 위한 설정
	font_name = matplotlib.font_manager.FontProperties(
            fname=font_filename
        ).get_name()
	font_filename = os.path.join(dirname, 'SpoqaHanSansRegular.ttf')  # 폰트 이름 바꾸기

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
  word_dict_array = list(
      map(lambda x: x.decode('utf-8'), sess.run('word_dict_array:0')))
  embeddings = sess.run('embeddings:0')

# start_getting_nearby_elements(5)
draw_plot(plot_num)