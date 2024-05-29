import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme()

import multiprocessing as mp

import string

def tokenize(text):
    for p in string.punctuation:
        text = text.replace(p, ' ')
    return text.lower().split()

class NBCtext():
    
    def fit(self, data):
                
        # sanity check
        if data.shape[1] != 2:
            return 'Wrong data format: data format is text/label'

        # train data
        train = data.copy()
        train.columns = ['text', 'labels']

        # tokenize
        train['tokens'] = train.text.apply(tokenize)
        
        # prior
        self.prior = {key: val.item() for key, val in train.labels.value_counts(normalize = True).to_frame().iterrows()}

        # dictionary
        #self.dict = {word: {key: 0 for key in self.prior.keys()} for word in list(set(train.tokens.sum()))}

        def initDict(tokens):
            for word in tokens:
                if word not in self.dict.keys(): self.dict[word] = {key: 0 for key in self.prior.keys()}

        self.dict = {}
        train.tokens.apply(initDict)

        def wordCounts(row):
            for word in row.tokens: self.dict[word][row.labels] += 1

        train.apply(wordCounts, axis = 1)

        # class totals
        self.totals = {key: sum([counts[key] for word, counts in self.dict.items()]) for key in self.prior.keys()} 

    def likelihood(self, word, key, alpha):
        
        if word in self.dict:
            return (self.dict[word][key] + alpha) /(self.totals[key] + len(self.dict) *alpha)
        else:
            return 1
            
    def predict(self, test, alpha = 1, soft = True):

        tokens = tokenize(test)
        posts = {key: val for key, val in self.prior.items()}

        for word in tokens:
            for key in posts:
                posts[key] *= self.likelihood(word, key, alpha = alpha)

        posts_sum = np.sum([val for val in posts.values()])
        for key, val in posts.items():
            posts[key] /= posts_sum

        if soft:
            return posts
        else:
            hard = [(k, v) for k, v in sorted(posts.items(), key = lambda item: item[1])][-2:]
            return hard[0][0] if hard[0][1] > hard[1][1] else (hard[1][0] if hard[1][1] > hard[0][1] else '??')

''' parallelized version '''

def worker_tokenize(data_chunk):
    data_chunk['tokens'] = data_chunk.text.apply(tokenize)
    return data_chunk


def worker_wordCounts(chunk_df, nbc_dict):

    def countWords(row):
        for word in row.tokens: nbc_dict[word][row.labels] += 1

    chunk_df.apply(countWords, axis = 1)

class NBCtext_p(NBCtext):
    
    def fit(self, data):
                
        # sanity check
        if data.shape[1] != 2:
            return 'Wrong data format: data format is text/label'

        # train data
        train = data.copy()
        train.columns = ['text', 'labels']

    def get_tokens(self):
        # tokenize
        with mp.Pool(processes = mp.cpu_count() -1) as pool:
            self.train = pd.concat(pool.map(worker_tokenize, np.array_split(train, mp.cpu_count() -1)), axis = 0)
        
    def get_prior(self):
        # prior
        self.prior = {key: val.item() for key, val in self.train.labels.value_counts(normalize = True).to_frame().iterrows()}

    def get_dict(self):
        # dictionary
        def initDict(tokens):
            for word in tokens:
                if word not in self.dict.keys(): self.dict[word] = {key: 0 for key in self.prior.keys()}

        self.dict = {}
        self.train.tokens.apply(initDict)

    def get_counts(self):

        def wordCounts(row):
            for word in row.tokens: self.dict[word][row.labels] += 1

        with mp.Pool(processes = mp.cpu_count() -1) as pool:
            #_ = pool.map(worker_wordCounts, [(data_chunk, self.dict) for data_chunk in np.array_split(self.train, mp.cpu_count() -1)])
            _ = [pool.apply(worker_wordCounts, args = (data_chunk, self.dict, )) for data_chunk in np.array_split(self.train, mp.cpu_count() -1)]

    def get_totals(self):
        # class totals
        self.totals = {key: sum([counts[key] for word, counts in self.dict.items()]) for key in self.prior.keys()} 
