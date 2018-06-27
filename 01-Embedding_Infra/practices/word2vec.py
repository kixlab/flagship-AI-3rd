# Word2Vec 모델을 간단하게 구현해봅니다.
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 이 파일과 같은 경로에 한글 폰트 파일을 넣어주세요
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'SpoqaHanSansRegular.ttf') # 폰트 이름 바꾸기

# matplot 에서 한글을 표시하기 위한 설정
font_name = matplotlib.font_manager.FontProperties(
    fname=filename
            ).get_name()
matplotlib.rc('font', family=font_name)

# 단어 벡터를 분석해볼 임의의 문장들
sentences = ["나 고양이 좋다",
             "나 강아지 좋다",
             "나 동물 좋다",
             "강아지 고양이 동물",
             "여자친구 고양이 강아지 좋다",
             "고양이 생선 우유 좋다",
             "강아지 생선 싫다 우유 좋다",
             "강아지 고양이 눈 좋다",
             "나 여자친구 좋다",
             "여자친구 나 싫다",
             "여자친구 나 영화 책 음악 좋다",
             "나 게임 만화 애니 좋다",
             "고양이 강아지 싫다",
             "강아지 고양이 좋다"]
# sentences = ["소두증을 유발한다고 알려진 지카바이러스가 남성의 생식능력도 저하시킬 수 있다는 동물 실험 결과가 나왔다",
# "1일 국제학술지 네이처에 따르면 미 워싱턴대 의대 연구진은 수컷 쥐가 지카바이러스에 감염되면 고환 크기가 눈에 띄게 작아지며 정자 수가 감소하고, 성 호르몬인 테스토스테론의 양도 줄어든다는 연구 결과를 발표했다", "지금까지 지카바이러스는 태아에게 소두증을 일으킨다고 알려져 태아와 여성의 생식기관 감염에 초점을 맞춘 연구들이 많았다", "소두증은 태아의 뇌가 다 자라지 않아 머리가 비정상적으로 작아지는 질환이다", "최근에는 브라질 등 남미 뿐 아니라 미국, 동남아시아에서도 환자가 발생하고 있다", "워싱턴대 연구진은 지카바이러스가 남성의 생식기관에 미치는 영향에 초점을 맞춰 연구를 진행했다", "연구진은 우선 수컷 쥐에게 지카바이러스를 감염시켰다", "1주가 지나자 생식기관인 고환에서도 바이러스가 발견됐다", "2주 뒤에는 수컷 쥐의 고환 크기가 현격하게 줄고 무게도 감소했다", "일반 쥐의 고환 무게는 75mg 이상이지만, 바이러스에 감염된 쥐의 경우 50mg도 되지 않았다", "3주 뒤 쥐의 고환 크기는 더욱 감소했으며 무게는 25mg 미만으로 줄어들었다", "연구진은 고환을 구성하는 세포가 죽었고, 고환 내부의 구조도 손상된 것을 확인했다",
#              "수컷의 핵심 생식기관인 고환이 지카바이러스의 공격으로 점차 기능을 잃어간 것이다", "고환은 생식세포인 정자와 성호르몬인 테스토스테론을 만드는 기관이다", "지카바이러스에 감염된 쥐는 고환의 크기가 작을 뿐 아니라 정자 수와 성호르몬 수치도 정상에 비해 적었다", "정자의 운동성도 현저히 감소했다", "오명돈 서울대병원 감염내과 교수는 “동물 모델을 이용해 지카바이러스가 정모세포(정자로 성장하는 세포), 정세관(정자가 나오는 작은 튜브) 세포 손상을 일으킨다는 것과 남성호르몬과 정자수, 고환크기, 가임력을 모두 감소시킨다는 것을 구체적으로 밝혔다”며 “다만 쥐 실험이므로 사람에게도 그대로 적용되는지는 알 수 없다”고 설명했다", "연구를 진행한 마이클 다이아몬드 교수도 “수컷 쥐에서 확인된 결과가 사람에게도 나타나는지는 아직 알지 못한다”며 “사람에게도 같은 영향이 있는지를 확인하기 위해선 추가 연구가 필요하다”고 밝혔다", "사람의 정자 속에서 지카바이러스가 발견된 적은 있다", "또 지카바이러스는 정액 속에서 수개월을 산다고 알려졌다", "이에 따라 세계보건기구(WHO)와 미국 질병예방통제센터(CDC)는 증상이 없더라도 지카 발생국가를 방문한 남성은 최소 6개월간 성관계 때 콘돔을 사용하라고 권고하고 있다"]

# 문장을 전부 합친 후 공백으로 단어들을 나누고 고유한 단어들로 리스트를 만듭니다.
word_sequence = " ".join(sentences).split()
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
# 문자열로 분석하는 것 보다, 숫자로 분석하는 것이 훨씬 용이하므로
# 리스트에서 문자들의 인덱스를 뽑아서 사용하기 위해,
# 이를 표현하기 위한 연관 배열과, 단어 리스트에서 단어를 참조 할 수 있는 인덱스 배열을 만듭합니다.
word_dict = {w: i for i, w in enumerate(word_list)}

# 윈도우 사이즈를 1 로 하는 skip-gram 모델을 만듭니다.
# 예) 나 게임 만화 애니 좋다
#   -> ([나, 만화], 게임), ([게임, 애니], 만화), ([만화, 좋다], 애니)
#   -> (게임, 나), (게임, 만화), (만화, 게임), (만화, 애니), (애니, 만화), (애니, 좋다)
skip_grams = []

for i in range(1, len(word_sequence) - 1):
    # (context, target) : ([target index - 1, target index + 1], target)
    # 스킵그램을 만든 후, 저장은 단어의 고유 번호(index)로 저장합니다
    target = word_dict[word_sequence[i]]
    context = [word_dict[word_sequence[i - 1]],
               word_dict[word_sequence[i + 1]]]

    # (target, context[0]), (target, context[1])..
    for w in context:
        skip_grams.append([target, w])


# skip-gram 데이터에서 무작위로 데이터를 뽑아 입력값과 출력값의 배치 데이터를 생성하는 함수
def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(data[i][0])  # target
        random_labels.append([data[i][1]])  # context word

    return random_inputs, random_labels


#########
# 옵션 설정
######
# 학습을 반복할 횟수
training_epoch = 300
# 학습률
learning_rate = 0.1
# 한 번에 학습할 데이터의 크기
batch_size = 20
# 단어 벡터를 구성할 임베딩 차원의 크기
# 이 예제에서는 x, y 그래프로 표현하기 쉽게 2 개의 값만 출력하도록 합니다.
embedding_size = 2
# word2vec 모델을 학습시키기 위한 nce_loss 함수에서 사용하기 위한 샘플링 크기
# batch_size 보다 작아야 합니다.
num_sampled = 15
# 총 단어 갯수
voc_size = len(word_list)


#########
# 신경망 모델 구성
######
inputs = tf.placeholder(tf.int32, shape=[batch_size])
# tf.nn.nce_loss 를 사용하려면 출력값을 이렇게 [batch_size, 1] 구성해야합니다.
labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

# word2vec 모델의 결과 값인 임베딩 벡터를 저장할 변수입니다.
# 총 단어 갯수와 임베딩 갯수를 크기로 하는 두 개의 차원을 갖습니다.
embeddings = tf.Variable(tf.random_uniform(
    [voc_size, embedding_size], -1.0, 1.0))
# 임베딩 벡터의 차원에서 학습할 입력값에 대한 행들을 뽑아옵니다.
# 예) embeddings     inputs    selected
#    [[1, 2, 3]  -> [2, 3] -> [[2, 3, 4]
#     [2, 3, 4]                [3, 4, 5]]
#     [3, 4, 5]
#     [4, 5, 6]]
selected_embed = tf.nn.embedding_lookup(embeddings, inputs)

# nce_loss 함수에서 사용할 변수들을 정의합니다.
nce_weights = tf.Variable(tf.random_uniform(
    [voc_size, embedding_size], -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([voc_size]))

# nce_loss 함수를 직접 구현하려면 매우 복잡하지만,
# 함수를 텐서플로우가 제공하므로 그냥 tf.nn.nce_loss 함수를 사용하기만 하면 됩니다.
loss = tf.reduce_mean(
    tf.nn.nce_loss(nce_weights, nce_biases, labels, selected_embed, num_sampled, voc_size))

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)


#########
# 신경망 모델 학습
######
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(1, training_epoch + 1):
        batch_inputs, batch_labels = random_batch(skip_grams, batch_size)

        _, loss_val = sess.run([train_op, loss],
                               feed_dict={inputs: batch_inputs,
                                          labels: batch_labels})

        if step % 10 == 0:
            print("loss at step ", step, ": ", loss_val)

    # matplot 으로 출력하여 시각적으로 확인해보기 위해
    # 임베딩 벡터의 결과 값을 계산하여 저장합니다.
    # with 구문 안에서는 sess.run 대신 간단히 eval() 함수를 사용할 수 있습니다.
    trained_embeddings = embeddings.eval()


#########
# 임베딩된 Word2Vec 결과 확인
# 결과는 해당 단어들이 얼마나 다른 단어와 인접해 있는지를 보여줍니다.
######
for i, label in enumerate(word_list):
    x, y = trained_embeddings[i]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2),
                 textcoords='offset points', ha='right', va='bottom')

plt.show()
