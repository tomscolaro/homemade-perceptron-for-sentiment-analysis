# models.py

from os import closerange
from sentiment_data import *
from utils import *

from collections import Counter
import numpy as np

import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")

class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        
        print('Using Unigram Feature Extractor.')
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        labels = []
        res = []
        counter_hold =[]

        if add_to_indexer:
            for i in sentence:
                labels.append(i.label)
                for j in i.words:
                    j_lower = re.sub(r'[^\w\s]','',j.lower())
                    if j_lower != '':
                       counter_hold.append(j_lower)

            counter_hold_res = Counter(counter_hold)

            for i in sentence:
                labels.append(i.label)
                for j in i.words:
                    j_lower = re.sub(r'[^\w\s]','',j.lower())
                    if counter_hold_res[j_lower] > 1  and counter_hold_res[j_lower] < 230:
                        self.indexer.add_and_get_index(j_lower)


            for i in sentence:
                hold = np.zeros(self.indexer.__len__()  + 1)
                for j in set(i.words):
                    j_lower = re.sub(r'[^\w\s]','',j.lower())
                    if self.indexer.contains(j_lower):
                            hold[self.indexer.index_of(j_lower)] = 1 
                hold[-1] = 1
                res.append(hold)
                

            
            return np.array(res), np.array(labels), self.indexer.__len__() + 1
        
        else:
            hold = np.zeros(self.indexer.__len__() + 1)
        
            for i in sentence:
                i_lower = re.sub(r'[^\w\s]','',i.lower())
                if self.indexer.contains(i_lower):
                    hold[self.indexer.index_of(i_lower)] = 1 
                    hold[-1] = 1

            return np.array(hold)

class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        
        print('Using Bigram Feature Extractor.')
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        stopword_list = stopwords.words('english')
        labels = []
        res = []
        bigram_holder = []
               #counter_hold = Counter()

        if add_to_indexer:
            
            for i in sentence:
                labels.append(i.label)
                words = []
                bigrams = []

                for j in i.words:
                    j_lower = re.sub(r'[^\w\s]','',j.lower())
                    if j_lower != '':
                       words.append(j_lower)
                
                for j in range(0, len(words) -1, 2):
                    bigram = words[j] + '|' + words[j+1]
                    bigrams.append(bigram)


                for j in bigrams:
                    self.indexer.add_and_get_index(j)


            for i in sentence:
                hold = np.zeros(self.indexer.__len__() + 1)
                words = []
                bigrams = []

                for j in i.words:
                    j_lower = re.sub(r'[^\w\s]','',j.lower())
                    if j_lower != '':
                       words.append(j_lower)
                
                for j in range(0, len(words) -1, 2):
                    bigram = words[j] + '|' + words[j+1]
                    bigrams.append(bigram)
                    #print(bigram)

                for j in bigrams:
                    hold[self.indexer.index_of(j)] = 1
                    hold[-1] = 1
                

                res.append(hold)
                
            return np.array(res), np.array(labels), self.indexer.__len__() + 1
        
        else:
            hold = np.zeros(self.indexer.__len__() + 1)
            words = []
            for i in sentence:
                i_lower = re.sub(r'[^\w\s]','',i.lower())
                if i_lower != '':
                    words.append(i_lower)

            for i in range(0, len(words) -1, 2):
                bigram = words[i] +'|'+ words[i+1]

                if self.indexer.contains(bigram):
                    hold[self.indexer.index_of(bigram)] = 1 
                    hold[-1] = 1

            return  np.array(hold)

class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        
        print('Using ~Better~ Feature Extractor.')
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        stopword_list = stopwords.words('english')
        labels = []
        res = []
        counter_hold =[]

        if add_to_indexer:
            for i in sentence:
                labels.append(i.label)
                for j in i.words:
                    j_lower = re.sub(r'[^\w\s]','',j.lower())
                    if j_lower != '' and j_lower not in stopword_list:
                       counter_hold.append(j_lower)

            counter_hold_res = Counter(counter_hold)

            for i in sentence:
                labels.append(i.label)
                for j in i.words:
                    j_lower = re.sub(r'[^\w\s]','',j.lower())
                    if counter_hold_res[j_lower] > 1  and counter_hold_res[j_lower] < 300:
                        self.indexer.add_and_get_index(j_lower)


            for i in sentence:
                hold = np.zeros(self.indexer.__len__()  + 1)
                for j in set(i.words):
                    j_lower = re.sub(r'[^\w\s]','',j.lower())
                    if self.indexer.contains(j_lower):
                            hold[self.indexer.index_of(j_lower)] = 1 
                hold[-1] = 1
                res.append(hold)
                
            return np.array(res), np.array(labels), self.indexer.__len__() + 1
        
        else:
            hold = np.zeros(self.indexer.__len__() + 1)
        
            for i in sentence:
                i_lower = re.sub(r'[^\w\s]','',i.lower())
                if self.indexer.contains(i_lower):
                    hold[self.indexer.index_of(i_lower)] = 1 
                    hold[-1] = 1

            return np.array(hold)

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")

class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1

#start with this
class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights, feature_processor):
        self.w = weights
        self.feat_extractor = feature_processor

    def predict(self, sentence: List[str]) -> int:
        x = self.feat_extractor.extract_features(sentence , add_to_indexer=False)
        return int(np.dot(self.w, x ) >= .5)

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights, feature_processor):
        self.w = weights
        self.feat_extractor = feature_processor

    def predict(self, sentence: List[str]) -> int:
        x = self.feat_extractor.extract_features(sentence , add_to_indexer=False)
        prob_positive = (np.exp(np.dot(self.w, x)))/(1 +np.exp(np.dot(self.w, x) )) 
        #prob_negative = (1)/(1 +np.e**np.dot(self.w, x) ) 
        
        return int(prob_positive >= .5)

#start with this
def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    epochs = 50
    x, y, feature_width = feat_extractor.extract_features(train_exs, add_to_indexer=True)
    step = .25

    w = np.random.rand(feature_width)
    randomize = np.arange(len(x))

    for e in range(0, epochs):
        #shuffling on each epoch
        np.random.shuffle(randomize)
        x = x[randomize]
        y = y[randomize]

        for i in range(0, x.shape[0]):
            y_pred = int(np.dot(w, x[i]) >= .5)

            if y_pred == int(y[i]):
                w = w
              
            else:
                if y[i] == 1:
                    w = w + step*x[i]
                if y[i] == 0:
                    w = w - step*x[i]

    return PerceptronClassifier(weights = w, feature_processor=feat_extractor )

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    #epochs 25, step .8
    epochs = 25

    x, y, feature_width = feat_extractor.extract_features(train_exs, add_to_indexer=True)
    step = .3
    
    w = np.random.rand(feature_width)
    #w = np.ones(feature_width)
    randomize = np.arange(len(x))
    
    for e in range(0, epochs):
        if e/epochs > .9:
            step *= .9

        np.random.shuffle(randomize)
        x = x[randomize]
        y = y[randomize]
        for i in range(0, x.shape[0]):
            prob_positive = (np.exp(np.dot(w, x[i])))/(1 +np.exp(np.dot(w, x[i]) )) 
            #prob_negative = (1)/(1 +np.e**np.dot(w, x[i]) ) 
           

            if y[i] == 1:
                w = w + step*x[i]*(1- prob_positive)
            else:
                w = w - step*x[i]*(prob_positive)

    return LogisticRegressionClassifier(weights = w, feature_processor=feat_extractor )


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model