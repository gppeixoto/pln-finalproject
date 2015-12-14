import Processor
import numpy as np
import cPickle as pickle
from nltk.corpus import stopwords
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
# from sklearn.
from sklearn.preprocessing import scale
from sklearn.svm import LinearSVC
import twokenize
from scipy.sparse import hstack

stop = stopwords.words("portuguese")
stop.remove(u'n\xe3o')
pr = Processor.Processor(stop)
corpus = pickle.load(open("pt.p", "r"))
normalized_corpus, twitterFeatures = pr.process(corpus, verbose=True)
to_keep = np.array([i for i, feats \
                        in enumerate(twitterFeatures)\
                        if feats[-1] in [-1, +1]])
normalized_corpus = [tweet for i, tweet\
                        in enumerate(normalized_corpus)\
                        if i in to_keep]
labels = np.array([i[-1] for i in twitterFeatures])
labels = labels[to_keep]
twitterFeatures = twitterFeatures[to_keep]
twitterFeatures = scale(np.array([i[:-1] for i in twitterFeatures]))

assert (twitterFeatures.shape[0] == len(normalized_corpus))

tweet_tokenizer = twokenize.tokenize
tfidf = Tfidf(ngram_range=(1,2), binary=True, tokenizer=tweet_tokenizer)
X = tfidf.fit_transform(normalized_corpus)

for tr, ts in KFold(n=len(normalized_corpus), n_folds=10):
    train = X[tr]
    test = X[ts]
    clf = LinearSVC()
    clf.fit(train, labels[tr])
    preds = clf.predict(test)
    acc1 = (preds == labels[ts]).sum() / (len(preds)+.0)
    train_f = twitterFeatures[tr]
    train = hstack([train, train_f])
    test_f = twitterFeatures[ts]
    test = hstack([test, test_f])
    clf.fit(train, labels[tr])
    preds = clf.predict(test)
    acc2 = (preds == labels[ts]).sum() / (len(preds)+.0)
    print 'Acc1: %.2f\t Acc2: %.2f' % (acc1, acc2)


