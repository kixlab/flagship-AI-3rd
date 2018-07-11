# Gensim Trainer v0.1

from gensim.models import Word2Vec

def load_sentence(filename):
	with open(filename, 'rb') as readfile:
		lines = [l.decode('utf8', 'ignore').strip().split(' ')
                    for l in readfile.readlines()]
	return lines


tokenized_contents = load_sentence('../../results/token-scripts-reduce3.txt')[:100000]
embedding_model = Word2Vec(tokenized_contents, size=50, window = 2, min_count=5, workers=4, iter=10000, sg=1)

embedding_model.save('../../results/token-scripts-reduce3-100000-word2vec')

print(embedding_model.most_similar(positive=["나는"], topn=100))
