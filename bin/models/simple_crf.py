import sklearn_crfsuite
from sklearn.externals import joblib

class SimpleCrf:
  def __init__(self):
    self.model = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

  def sent2features(self, sent):
    s = sent.split(" ")
    return [self.word2features(s[:-3], i) for i in range(len(s[:-3]))]

  def fit(self, X, y):
    train_X = [self.sent2features(s) for s in X]
    self.model.fit(train_X, y)

  def predict(self, X):
    return self.model.predict(self.sent2features(X))

  def save(self, path):
    joblib.dump(self.model, path)

  def word2features(self, sent, i):
    word, postag = sent[i].split('/')

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1, postag1 = sent[i-1].split('/')
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1, postag1 = sent[i].split('/')
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features
