import copy
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.preprocessing import scale
import numpy as np
import re
import cPickle as pickle
from nltk.corpus import stopwords
import twokenize # tweet tokenizer
import time
from scipy.sparse import csr_matrix as toSparse
from scipy.sparse import hstack
from warnings import warn
from sys import stdout

tweet_tokenizer = twokenize.tokenize
# Taking important words off of the stopwords dictionary

adverbs = set(['muito', 'extremamente', 'bastante'])

emojiList = set([':-)', '(-:', '=)', '(=', '(:', ':)', ':-(', ')-:', '=(', ')=', ':(', '):', ':D', '^_^', '^__^', '^___^', ':d', 'd:', \
    ': )', '( :', ': (', ') :', '8)', '(8', '8(', ')8', '8 )', ') 8', '8 (', ';)', '(;', '; )', '( ;', ';-)', '(-;'])

posEmoji = set([':-)', '(-:', '=)', '(=', '(:', ':)', ':-(', ':D', '^_^', '^__^', '^___^', ':d', 'd:', ': )', '( :', '8)', \
            '(8', '8 )', ';)', '; )', '; )', '( ;', ';-)', '(-;', '(;'])

negEmoji = emojiList.difference(posEmoji)

punctuation = set([',', ';', '.', ':', '.', '!', '?', '\"', '*', '\'', '(', ')', '-'])
pattern = re.compile(r'(.)\1{2,}', re.DOTALL) # for elongated words truncation

class Processor:
    def __init__(self, stopwords, tokenizer=tweet_tokenizer, ngrams=2):
        self.tokenizer = tokenizer
        self.stopwords = stopwords
        self.__target_not = u'n\xe3o'
        for adv in adverbs: 
            if adv in self.stopwords: self.stopwords.remove(adv)
            #TODO: use the adverbs below to generate lexicons in target language
        self.__adverbs = adverbs
        self.lang = 'pt'
        self.__fitted = False
        # WARNING: do NOT change the parameters of the vectorization. It is already
        # set to the optimal configuration.
        self.__vectorizer = Tfidf(ngram_range=(1,ngrams), binary=True,
            tokenizer=self.tokenizer)

    def __preprocess(self, tweetList, verbose=False):
        rep_count = []
        hst_count = []
        hst_last = []
        exc_count = []
        exc_last = []
        qst_count = []
        qst_last = []
        neg_count = []
        tw_length = [] 
        labels = []
        ll = len(tweetList)
        dot = ll / 50
        for x in xrange(ll):
            if verbose and dot > 0 and x % dot == 0:
                stdout.write("."); stdout.flush()
            tweet = tweetList[x].lower().encode('utf-8').decode('utf-8')

            # Count reps
            reps = pattern.findall(tweet)
            if reps != []: tweet = pattern.sub(r'\1\1', tweet)
            rep_count.append(len(reps))

            # Tokenizing
            tweet = self.tokenizer(tweet) # ok to use independent of language

            # Removing stopwords and retweet noise
            tweet = [word for word in tweet if word not in self.stopwords and not word.startswith('RT')]

            # Normalizing mentions, hyperlinks
            reps = 0. # float is intended type
            hsts = 0. # necessary for scaling
            excs = 0.
            qsts = 0.
            negs = 0.
            last = -1.
            label = np.inf
            for i, word in enumerate(tweet):
                if word.startswith(('.@', '@')): #mention
                    tweet[i] = '___mention___'
                if word.startswith(('www','http')):
                    tweet[i] = '___url___'
                if word.startswith('!'):
                    excs += 1
                    last = 0
                if word.startswith('?'): #TODO: problem with ?!, !?, account for this
                    qsts += 1
                    last = 1
                if word.startswith('#'):
                    hsts += 1
                    last = 2
                if word == self.__target_not:
                    negs += 1
                    tweet[i] = ''
                    if (i+1)<len(tweet):
                        tweet[i+1] = self.__target_not+'___'+tweet[i+1]
                    else:
                        tweet[i] = self.__target_not
                if label == np.inf and word in posEmoji:
                    label = +1
                elif label == np.inf and word in negEmoji:
                    label = -1
            hst_count.append(hsts)
            qst_count.append(qsts)
            exc_count.append(excs)
            neg_count.append(negs)
            tw_length.append(len(tweet))
            labels.append(label)
            # Removing punctuation
            tweet = [''.join([w for w in word if w not in punctuation]) for word in tweet if len(word)>2]
            tweet = ' '.join(tweet) 
            tweetList[x] = tweet
        return (tweetList, rep_count, hst_count, exc_count, qst_count, neg_count, tw_length, labels)

    def process(self, tweetList, verbose=False):
        ll = copy.deepcopy(tweetList)
        t0 = time.time()
        if verbose:
            print 'Normalizing and extracting features'
        ret = self.__preprocess(ll, verbose)
        corpus = ret[0]
        rep_count, hst_count, exc_count, qst_count, neg_count, \
            tw_length, labels = map(lambda x: np.array(x), list(ret[1:]))
        feats = np.vstack((rep_count, hst_count, exc_count, qst_count, \
                            neg_count, tw_length, labels)).transpose()
        if verbose:
            print '\nTime elapsed on processing and feature extraction: %.0fs' % ((time.time()-t0))
        return (corpus, feats)