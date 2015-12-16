import Processor
import numpy as np
import cPickle as pickle
from nltk.corpus import stopwords
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.preprocessing import scale
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression as MaxEnt
from sklearn.naive_bayes import MultinomialNB as NB
import twokenize
import time
from scipy.sparse import hstack

t0 = time.time()
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


accs = []
ps = []
rs = []
fs = []

print '#'*40
print 'NO Twitter Features'
print 'SVM - Linear Kernel'
print '#'*40
print 'ACC\tPR\tRE\tF1'
print '#'*40
i=1
for tr, ts in KFold(n=len(normalized_corpus), n_folds=10):
    train = X[tr]
    test = X[ts]
    clf = LinearSVC()
    clf.fit(train, labels[tr])
    ytrue = labels[ts]
    ypred = clf.predict(test)
    acc = (ytrue == ypred).sum() / (len(ypred)+.0)
    p, r, f, s = prfs(ytrue, ypred, average='binary')
    accs.append(acc)
    ps.append(p)
    rs.append(r)
    fs.append(f)
    print "%.2f\t%.2f\t%.2f\t%.2f KFoldRnd%d" % (acc,p,r,f,i)
    i += 1
print '#'*40
print 'Mean accuracy: %.2f' % (np.mean(accs))
print 'Mean precision: %.2f' % (np.mean(ps))
print 'Mean recal: %.2f' % (np.mean(rs))
print 'Mean f-score: %.2f' % (np.mean(fs))


accs = []
ps = []
rs = []
fs = []

print '#'*40
print 'Using Twitter Features'
print 'SVM - Linear Kernel'
print '#'*40
print 'ACC\tPR\tRE\tF1'
print '#'*40
i=1
for tr, ts in KFold(n=len(normalized_corpus), n_folds=10):
    train = X[tr]
    train_f = twitterFeatures[tr]
    train = hstack([train, train_f])

    test = X[ts]
    test_f = twitterFeatures[ts]
    test = hstack([test, test_f])

    clf = LinearSVC()
    clf.fit(train, labels[tr])
    ytrue = labels[ts]
    ypred = clf.predict(test)
    acc = (ytrue == ypred).sum() / (len(ypred)+.0)
    p, r, f, s = prfs(ytrue, ypred, average='binary')
    accs.append(acc)
    ps.append(p)
    rs.append(r)
    fs.append(f)
    print "%.2f\t%.2f\t%.2f\t%.2f KFoldRnd%d" % (acc,p,r,f,i)
    i += 1
print '#'*40
print 'Mean accuracy: %.2f' % (np.mean(accs))
print 'Mean precision: %.2f' % (np.mean(ps))
print 'Mean recal: %.2f' % (np.mean(rs))
print 'Mean f-score: %.2f' % (np.mean(fs))

print '#'*40
print 'NO Twitter Features'
print 'SVM - RBF Kernel'
print '#'*40
print 'ACC\tPR\tRE\tF1'
print '#'*40
i=1
for tr, ts in KFold(n=len(normalized_corpus), n_folds=10):
    train = X[tr]
    test = X[ts]
    clf = SVC()
    clf.fit(train, labels[tr])
    ytrue = labels[ts]
    ypred = clf.predict(test)
    acc = (ytrue == ypred).sum() / (len(ypred)+.0)
    p, r, f, s = prfs(ytrue, ypred, average='binary')
    accs.append(acc)
    ps.append(p)
    rs.append(r)
    fs.append(f)
    print "%.2f\t%.2f\t%.2f\t%.2f KFoldRnd%d" % (acc,p,r,f,i)
    i += 1

print '#'*40
print 'Mean accuracy: %.2f' % (np.mean(accs))
print 'Mean precision: %.2f' % (np.mean(ps))
print 'Mean recal: %.2f' % (np.mean(rs))
print 'Mean f-score: %.2f' % (np.mean(fs))


accs = []
ps = []
rs = []
fs = []

print '#'*40
print 'Using Twitter Features'
print 'SVM - RBF Kernel'
print '#'*40
print 'ACC\tPR\tRE\tF1'
print '#'*40
i=1
for tr, ts in KFold(n=len(normalized_corpus), n_folds=10):
    train = X[tr]
    train_f = twitterFeatures[tr]
    train = hstack([train, train_f])

    test = X[ts]
    test_f = twitterFeatures[ts]
    test = hstack([test, test_f])

    clf = SVC()
    clf.fit(train, labels[tr])
    ytrue = labels[ts]
    ypred = clf.predict(test)
    acc = (ytrue == ypred).sum() / (len(ypred)+.0)
    p, r, f, s = prfs(ytrue, ypred, average='binary')
    accs.append(acc)
    ps.append(p)
    rs.append(r)
    fs.append(f)
    print "%.2f\t%.2f\t%.2f\t%.2f KFoldRnd%d" % (acc,p,r,f,i)
    i += 1
print '#'*40
print 'Mean accuracy: %.2f' % (np.mean(accs))
print 'Mean precision: %.2f' % (np.mean(ps))
print 'Mean recal: %.2f' % (np.mean(rs))
print 'Mean f-score: %.2f' % (np.mean(fs))

print '#'*40
print 'NO Twitter Features'
print 'SVM - Sigmoid Kernel'
print '#'*40
print 'ACC\tPR\tRE\tF1'
print '#'*40
i=1
for tr, ts in KFold(n=len(normalized_corpus), n_folds=10):
    train = X[tr]
    test = X[ts]
    clf = SVC(kernel = 'sigmoid')
    clf.fit(train, labels[tr])
    ytrue = labels[ts]
    ypred = clf.predict(test)
    acc = (ytrue == ypred).sum() / (len(ypred)+.0)
    p, r, f, s = prfs(ytrue, ypred, average='binary')
    accs.append(acc)
    ps.append(p)
    rs.append(r)
    fs.append(f)
    print "%.2f\t%.2f\t%.2f\t%.2f KFoldRnd%d" % (acc,p,r,f,i)
    i += 1

print '#'*40
print 'Mean accuracy: %.2f' % (np.mean(accs))
print 'Mean precision: %.2f' % (np.mean(ps))
print 'Mean recal: %.2f' % (np.mean(rs))
print 'Mean f-score: %.2f' % (np.mean(fs))


accs = []
ps = []
rs = []
fs = []

print '#'*40
print 'Using Twitter Features'
print 'SVM - Sigmoid Kernel'
print '#'*40
print 'ACC\tPR\tRE\tF1'
print '#'*40
i=1
for tr, ts in KFold(n=len(normalized_corpus), n_folds=10):
    train = X[tr]
    train_f = twitterFeatures[tr]
    train = hstack([train, train_f])

    test = X[ts]
    test_f = twitterFeatures[ts]
    test = hstack([test, test_f])

    clf = SVC(kernel = 'sigmoid')
    clf.fit(train, labels[tr])
    ytrue = labels[ts]
    ypred = clf.predict(test)
    acc = (ytrue == ypred).sum() / (len(ypred)+.0)
    p, r, f, s = prfs(ytrue, ypred, average='binary')
    accs.append(acc)
    ps.append(p)
    rs.append(r)
    fs.append(f)
    print "%.2f\t%.2f\t%.2f\t%.2f KFoldRnd%d" % (acc,p,r,f,i)
    i += 1

print '#'*40
print 'Mean accuracy: %.2f' % (np.mean(accs))
print 'Mean precision: %.2f' % (np.mean(ps))
print 'Mean recal: %.2f' % (np.mean(rs))
print 'Mean f-score: %.2f' % (np.mean(fs))

print '#'*40
print 'NO Twitter Features'
print 'MaxEnt'
print '#'*40
print 'ACC\tPR\tRE\tF1'
print '#'*40
i=1
for tr, ts in KFold(n=len(normalized_corpus), n_folds=10):
    train = X[tr]
    test = X[ts]
    clf = MaxEnt()
    clf.fit(train, labels[tr])
    ytrue = labels[ts]
    ypred = clf.predict(test)
    acc = (ytrue == ypred).sum() / (len(ypred)+.0)
    p, r, f, s = prfs(ytrue, ypred, average='binary')
    accs.append(acc)
    ps.append(p)
    rs.append(r)
    fs.append(f)
    print "%.2f\t%.2f\t%.2f\t%.2f KFoldRnd%d" % (acc,p,r,f,i)
    i += 1

print '#'*40
print 'Mean accuracy: %.2f' % (np.mean(accs))
print 'Mean precision: %.2f' % (np.mean(ps))
print 'Mean recal: %.2f' % (np.mean(rs))
print 'Mean f-score: %.2f' % (np.mean(fs))


accs = []
ps = []
rs = []
fs = []

print '#'*40
print 'Using Twitter Features'
print 'MaxEnt'
print '#'*40
print 'ACC\tPR\tRE\tF1'
print '#'*40
i=1
for tr, ts in KFold(n=len(normalized_corpus), n_folds=10):
    train = X[tr]
    train_f = twitterFeatures[tr]
    train = hstack([train, train_f])

    test = X[ts]
    test_f = twitterFeatures[ts]
    test = hstack([test, test_f])

    clf = MaxEnt()
    clf.fit(train, labels[tr])
    ytrue = labels[ts]
    ypred = clf.predict(test)
    acc = (ytrue == ypred).sum() / (len(ypred)+.0)
    p, r, f, s = prfs(ytrue, ypred, average='binary')
    accs.append(acc)
    ps.append(p)
    rs.append(r)
    fs.append(f)
    print "%.2f\t%.2f\t%.2f\t%.2f KFoldRnd%d" % (acc,p,r,f,i)
    i += 1
print '#'*40
print 'Mean accuracy: %.2f' % (np.mean(accs))
print 'Mean precision: %.2f' % (np.mean(ps))
print 'Mean recal: %.2f' % (np.mean(rs))
print 'Mean f-score: %.2f' % (np.mean(fs))

print '#'*40
print 'NO Twitter Features'
print 'Naive Bayes'
print '#'*40
print 'ACC\tPR\tRE\tF1'
print '#'*40
i=1
for tr, ts in KFold(n=len(normalized_corpus), n_folds=10):
    train = X[tr]
    test = X[ts]
    clf = NB()
    clf.fit(train, labels[tr])
    ytrue = labels[ts]
    ypred = clf.predict(test)
    acc = (ytrue == ypred).sum() / (len(ypred)+.0)
    p, r, f, s = prfs(ytrue, ypred, average='binary')
    accs.append(acc)
    ps.append(p)
    rs.append(r)
    fs.append(f)
    print "%.2f\t%.2f\t%.2f\t%.2f KFoldRnd%d" % (acc,p,r,f,i)
    i += 1

print '#'*40
print 'Mean accuracy: %.2f' % (np.mean(accs))
print 'Mean precision: %.2f' % (np.mean(ps))
print 'Mean recal: %.2f' % (np.mean(rs))
print 'Mean f-score: %.2f' % (np.mean(fs))

print '\n\nOverall execution time: %.0fs' % ((time.time()-t0)) 