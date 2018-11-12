from models.simple_crf import SimpleCrf
from utils.logger import Logger
from utils.file import load_text_file
from sklearn.metrics import classification_report, accuracy_score

input_X = '../data-da/swda-data-pos_X.csv'
input_y = '../data-da/swda-data-pos_Y.csv'
model_path = '../models/181105-simple-pos.pkl'

train_rate = .85

my_labels = ['d', 'e', 'k', 'x']
my_tags = ['Disclosure', 'Statement-non-opinion',
           'Acknowledge', 'Action-directive', 'Commits', 'Rest']

logger = Logger('swda-simple-runner')

X = list(filter(lambda x: x is not '', load_text_file(input_X)))
y = list(filter(lambda x: x is not '', load_text_file(input_y)))

train_index = int(len(X) * .85)
X_train = X[:train_index]
y_train = y[:train_index]
X_test = X[train_index:]
y_test = y[train_index:]
print(X_train[0])

model = SimpleCrf()
model.fit(X_train, y_train)
model.save(model_path)

y_pred = model.predict(X_test)
logger.write(f'<{model_path}>')
logger.write('accuracy %s' % accuracy_score(y_pred, y_test))
logger.write(classification_report(
    y_test, y_pred, labels=my_labels, target_names=my_tags))
