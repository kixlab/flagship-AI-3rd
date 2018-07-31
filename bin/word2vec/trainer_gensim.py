# Gensim Trainer v0.1.3

from tqdm import trange, tqdm
from gensim.models import Word2Vec
import _uu as uu

## Variables
input_file = '../../results/180731-padded-500000-splited.txt'
output_file = '../../results/180731-padded-500000sent'
log_file = '../../logs/180731-padded-500000sent'

training_epoch = 50000
max_sentences = 500000
log_per = 1000
save_per = 5000


## Functions
def load_sentence(filename):
	with open(filename, 'rb') as readfile:
		lines = [l.decode('utf8', 'ignore').strip().split(' ')
                    for l in readfile.readlines()]
	return lines


## Main
logger = uu.get_custom_logger('trainer_gensim', log_file)

logger.info("Load sentences...")
tokenized_contents = load_sentence(input_file)[:max_sentences]

logger.info("Create Word2Vec model...")
model = Word2Vec(iter=1, size=50, window=2, min_count=5,
                 workers=4, sg=1, compute_loss=True)

logger.info("Build Vocab...")
model.build_vocab(tokenized_contents)

loss_sum = 0
for epoch in trange(1, training_epoch + 1):
	model.train(
		tokenized_contents, total_examples=model.corpus_count, epochs=model.epochs, compute_loss=True)
	loss_sum += model.get_latest_training_loss()

	if log_per > 0 and epoch % log_per == 0:
		logger.debug('Loss from %6d to %6d : %f' %(epoch - log_per, epoch, loss_sum / log_per))
		loss_sum = 0
	
	if epoch % save_per == 0:
		model.save(output_file + '_%d' % epoch)
		logger.info("Model with %6d iterations is saved." % epoch)

model.save(output_file + '_final')
logger.info('Final Model is saved!')
