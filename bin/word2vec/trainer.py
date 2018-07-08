# Seongho's word2vec traniner v0.1
# Code From globin's TensorFlow Tutorials
# https://github.com/golbin/TensorFlow-Tutorials/blob/master/04%20-%20Neural%20Network%20Basic/03%20-%20Word2Vec.py

# Word2Vec 모델을 간단하게 구현해봅니다.
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import time
import datetime
from sklearn.manifold import TSNE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## VARIABLES
IS_COLAB = False

FILE_NAME = '../results/token-scripts-plain-100000.txt'

word_list_file = '../../results/token-scripts-reduce3-word-list.txt'
skip_grams_file = '../../results/token-scripts-reduce3-skipgram.txt'

SAVE_MODEL_NAME = 'model/scripts-reduce3-100000sent'
WORD_COUNT = 2
LOSS_LOG_PER = 100
SAVING_MODEL_PER = 500

training_sentences = 100000
start_sentence = 0

# 학습을 반복할 횟수
training_epoch = 3000
# 학습률
learning_rate = 0.025
# 한 번에 학습할 데이터의 크기
batch_size = 2000
# 단어 벡터를 구성할 임베딩 차원의 크기
# 이 예제에서는 x, y 그래프로 표현하기 쉽게 2 개의 값만 출력하도록 합니다.
# normally 50 or 200 ~ 300 (may it depends on the vocab size)
embedding_size = 50
# word2vec 모델을 학습시키기 위한 nce_loss 함수에서 사용하기 위한 샘플링 크기
# batch_size 보다 작아야 합니다.
num_sampled = 1500

## Functions
def get_current_datetime():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def import_file(file_name):
    print("[%s] Import file..." % get_current_datetime())
    with open(file_name, 'rb') as f:
        sentences = [l.decode('utf8', 'ignore')
                     for l in f.readlines()][start_sentence:training_sentences]

    print("[%s] Create word list..." % get_current_datetime())
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    # 문자열로 분석하는 것 보다, 숫자로 분석하는 것이 훨씬 용이하므로
    # 리스트에서 문자들의 인덱스를 뽑아서 사용하기 위해,
    # 이를 표현하기 위한 연관 배열과, 단어 리스트에서 단어를 참조 할 수 있는 인덱스 배열을 만듭합니다.
    word_dict = {w: i for i, w in enumerate(word_list)}
    word_dict_reverse = dict(zip(word_dict.values(), word_dict.keys()))

    # Create skip-gram
    print("[%s] Create skip-grams..." % get_current_datetime())
    skip_grams = []
    for s in sentences:
        words = s.strip().split(' ')
        for i, word in enumerate(words):
            target = word_dict[word]
            context = []
            start_index = (i - WORD_COUNT) if (i > WORD_COUNT) else 0
            for close_word in words[start_index:(i + WORD_COUNT)]:
                context.append(word_dict[close_word])

            # (target, context[0]), (target, context[1])..
            for w in context:
                skip_grams.append([target, w])

    return word_list, word_dict, word_dict_reverse, skip_grams

def load_word_list(filename):
    with open(filename, 'rb') as readfile:
	    words = [l.decode('utf8', 'ignore').strip() for l in readfile.readlines()]
    return words

def load_skip_grams(filename):
    with open(filename, 'r') as readfile:
        sentences = readfile.read().strip().split(os.linesep)

    skip_grams = []
    for s in sentences:
        skip_grams.append(s.strip().split(' '))
    
    return skip_grams

# skip-gram 데이터에서 무작위로 데이터를 뽑아 입력값과 출력값의 배치 데이터를 생성하는 함수
def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(data[i][0])  # target
        random_labels.append([data[i][1]])  # context word

    return random_inputs, random_labels


## Main

if (not IS_COLAB):
	# 파일 경로 설정
	dirname = os.path.dirname(__file__)
	FILE_NAME = os.path.join(os.path.join(dirname, os.pardir), 'dataset/' + FILE_NAME)

# word_list, word_dict, word_dict_reverse, skip_grams = import_file(FILE_NAME)
word_list = load_word_list(word_list_file)
skip_grams = load_skip_grams(skip_grams_file)

# 윈도우 사이즈를 1 로 하는 skip-gram 모델을 만듭니다.
# 예) 나 게임 만화 애니 좋다
#   -> ([나, 만화], 게임), ([게임, 애니], 만화), ([만화, 좋다], 애니)
#   -> (게임, 나), (게임, 만화), (만화, 게임), (만화, 애니), (애니, 만화), (애니, 좋다)
# skip_grams = []

# for i in range(0, len(word_sequence)):
#     # 스킵그램을 만든 후, 저장은 단어의 고유 번호(index)로 저장합니다
#     target = word_dict[word_sequence[i]]
#     context = []
#     start_index = (i - WORD_COUNT) if (i > WORD_COUNT) else 0
#     for word in word_sequence[start_index:(i + WORD_COUNT)]:
#         context.append(word_dict[word])

#     # (target, context[0]), (target, context[1])..
#     for w in context:
#         skip_grams.append([target, w])

#########
# 옵션 설정
######
# 다른 설정은 최상단 VARIABLES에 포함되어 있습니다.
# 총 단어 갯수
voc_size = len(word_list)


#########
# 신경망 모델 구성
######
# Save variables
# tf.Variable(word_list, name="word_list")


inputs = tf.placeholder(tf.int32, shape=[batch_size])
# tf.nn.nce_loss 를 사용하려면 출력값을 이렇게 [batch_size, 1] 구성해야합니다.
labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

# word2vec 모델의 결과 값인 임베딩 벡터를 저장할 변수입니다.
# 총 단어 갯수와 임베딩 갯수를 크기로 하는 두 개의 차원을 갖습니다.
embeddings = tf.Variable(tf.random_uniform(
    [voc_size, embedding_size], -1.0, 1.0), name="embeddings")
# 임베딩 벡터의 차원에서 학습할 입력값에 대한 행들을 뽑아옵니다.
# 예) embeddings     inputs    selected
#    [[1, 2, 3]  -> [2, 3] -> [[2, 3, 4]
#     [2, 3, 4]                [3, 4, 5]]
#     [3, 4, 5]
#     [4, 5, 6]]
selected_embed = tf.nn.embedding_lookup(embeddings, inputs)

# nce_loss 함수에서 사용할 변수들을 정의합니다.
nce_weights = tf.Variable(tf.random_uniform(
    [voc_size, embedding_size], -1.0, 1.0), name="nce_weights")
nce_biases = tf.Variable(tf.zeros([voc_size]), name="nce_biases")

with tf.device("/gpu:0"):
    # nce_loss 함수를 직접 구현하려면 매우 복잡하지만,
    # 함수를 텐서플로우가 제공하므로 그냥 tf.nn.nce_loss 함수를 사용하기만 하면 됩니다.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(nce_weights, nce_biases, labels, selected_embed, num_sampled, voc_size))

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)



#########
# 신경망 모델 학습
######
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver()

    total_loss = 0
    print("[%s] Training Start!" % get_current_datetime())
    for step in range(1, training_epoch + 1):
        batch_inputs, batch_labels = random_batch(skip_grams, batch_size)

        _, loss_val = sess.run([train_op, loss],
                               feed_dict={inputs: batch_inputs,
                                          labels: batch_labels})

        total_loss += loss_val
        if step % LOSS_LOG_PER == 0:
            # print("loss from", (step - LOSS_LOG_PER), " step ", step, ": ", (total_loss / LOSS_LOG_PER))
            print("[%s] loss from step %6d to %6d : %f" % (get_current_datetime(), (step - LOSS_LOG_PER), step, (total_loss / LOSS_LOG_PER)))
            total_loss = 0

        if step % SAVING_MODEL_PER == 0:
            saver.save(sess, SAVE_MODEL_NAME, global_step=step)
            print("[%s] Model with %d iterations is saved." % (get_current_datetime(), step))

    saver.save(sess, SAVE_MODEL_NAME + "_final")
    print("[%s] Final model saved!" % get_current_datetime())

    
    # matplot 으로 출력하여 시각적으로 확인해보기 위해
    # 임베딩 벡터의 결과 값을 계산하여 저장합니다.
    # with 구문 안에서는 sess.run 대신 간단히 eval() 함수를 사용할 수 있습니다.
    trained_embeddings = embeddings.eval()
