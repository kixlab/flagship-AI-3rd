# https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from utils.file import read_json
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from utils.logger import Logger

train_file = "../data-da/swda-set-train.json"
test_file = "../data-da/swda-set-test.json"

logger = Logger('swda-simple-runner')

my_labels = ['d', 's', 'k', 'a', 'c', 'x']
my_tags = ['Disclosure', 'Statement-non-opinion', 'Acknowledge', 'Action-directive', 'Commits', 'Rest']

def split_x_y(data, x_name, y_name):
  X = []
  Y = []
  for d in data:
    X.append(' '.join(d[x_name]))
    Y.append(' '.join(d[y_name]))
  return X, Y

train_data = read_json(train_file)
X_train, y_train = split_x_y(train_data, 'tokens', 'tag')
test_data = read_json(test_file)
X_test, y_test = split_x_y(test_data, 'tokens', 'tag')

# NB
# md = Pipeline([('vect', CountVectorizer()),
#                ('tfidf', TfidfTransformer()),
#                ('clf', MultinomialNB()),
#                ])

# SVM
# md = Pipeline([('vect', CountVectorizer()),
#                 ('tfidf', TfidfTransformer()),
#                 ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
#                ])

# Logistic Regression
md = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', LogisticRegression(n_jobs=1, C=1e5)),
                   ])


md.fit(X_train, y_train)

y_pred = md.predict(X_test)

logger.write('<Logistic Regression model>')
logger.write('accuracy %s' % accuracy_score(y_pred, y_test))
logger.write(classification_report(y_test, y_pred, labels=my_labels, target_names=my_tags))
