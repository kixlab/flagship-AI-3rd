# Gensim Trainer v0.1.1

IS_COLAB = False

if IS_COLAB:
	import universal_utils as uu
else:
	import _uu as uu

## Variables
input_file = '../../results/token-scripts-reduce3.txt'
output_file = '../../results/gensim-test'

training_epoch = 1000
max_sentences = 100000
log_per = 50
save_per = 200

from gensim.models import Word2Vec

def load_sentence(filename):
	with open(filename, 'rb') as readfile:
		lines = [l.decode('utf8', 'ignore').strip().split(' ')
                    for l in readfile.readlines()]
	return lines

uu.print_dt("Load sentences...")
tokenized_contents = load_sentence(input_file)[:max_sentences]

uu.print_dt("Create Word2Vec model...")
model = Word2Vec(iter=1, size=50, window=2, min_count=5,
                 workers=4, sg=1, compute_loss=True)

uu.print_dt("Build Vocab...")
model.build_vocab(tokenized_contents)

loss_sum = 0
for epoch in range(1, training_epoch + 1):
	model.train(
		tokenized_contents, total_examples=model.corpus_count, epochs=model.epochs, compute_loss=True)
	loss_sum += model.get_latest_training_loss()

	if epoch % log_per == 0:
		uu.print_dt('Loss from %6d to %6d : %f' %(epoch - log_per, epoch, loss_sum / log_per))
		loss_sum = 0
	
	if epoch % save_per == 0:
		model.save(output_file + '_%d' % epoch)
		uu.print_dt("Model with %6d iterations is saved." % epoch)

model.save(output_file + '_final')
uu.print_dt('Final Model is saved!')
