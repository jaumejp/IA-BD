import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
sns.set_theme()
import string
import random

def print_first_n_elements(dictionary, n):
    count = 0
    msg = ""
    for key, value in dictionary.items():
        msg += f"{key}: {value} \n"
        count += 1
        if count == n:
            break
    return msg

class NaiveBayes():

    def __init__(self, train):
        df = self.add_tokenize_col(train)
        self.init_dictionary(df)
        self.fill_dictionary(df)
        self.set_n_words_for_label(df)
        self.set_prior(df)

    # Representation of the object when using the print() function.
    def __repr__(self):
        return '+++ head of test data: \n%s' %(print_first_n_elements(self.dictionary, 3))

    # print the n, first registers of the dictionary
    def head(self, n=5):
        print('+++ first %d elements of test data: \n%s' %(n, print_first_n_elements(self.dictionary, n)))

    # Add the column tokenize to the dataframe
    def add_tokenize_col(self, df):
        df['tokens'] = df.Text.apply(self.tokenize)
        return df

    # Given a message (a group of words), this function returns an array containing each word separately.
    # It processes the message by replacing punctuation with spaces, converting to lowercase, and then splitting into individual words.
    def tokenize(self, message):
        for p in string.punctuation:
            message = message.replace(p, ' ')
        return message.lower().split()

    # Create a dictionary where each unique word is a key, and the corresponding value is another dictionary.
    # The inner dictionary contains all possible labels from the 'Label' column in the DataFrame,
    # and each label is initialized with a counter set to 0.
    def init_dictionary(self, df):
        self.dictionary = { word: {label: 0 for label in df.Label.unique()} for word in list(set(df['tokens'].sum())) }

    # Check all 'tokens' columns in every row. Every time a word appears, increase the counter for that word
    # in the current 'label' being processed.
    def fill_dictionary(self, df):
        _ = df.apply(self.count_words, axis = 1)
        del _

    def count_words(self, message):
        for word in message.tokens:
            self.dictionary[word][message.Label] +=1

    # We aim to count the occurrences of each word or token for a specific label ('ham' or 'spam', for example).
    # Iterate through all key-value pairs in the dictionary. The key is 'word' and the value is 'counts'
    # The value or 'counts' is another dictionary representing counts for each label. Ex: 'love' : {'ham': 0, 'spam': 25}.
    # Access all the words and retrieve their counts in the inner dictionary with the specified label to create a list.
    # Finally, apply the sum() function to calculate the total occurrences of a word with the specified label.
    # So, we do it for all the possible labels
    def set_n_words_for_label(self, df):
        self.n_words_for_label = { f'n_{label}': sum(counts[label] for word, counts in self.dictionary.items()) for label in df.Label.unique() }

    # Set all the priors (how many times appears each label in the dataset)
    def set_prior(self, df):
        self.prior = df.Label.value_counts(normalize = True)

    # Given new data, overrides all the train data given on the __init__ method.
    def fit(self, new_train_data):
        self.__init__(new_train_data)

    # Show histogram of frequency of words
    def hist_word_count_plt(self, col='', n=100):
        if col == '':
            col = next(iter(next(iter(self.dictionary.items()))[1].items()))[0]
        try:
            if int(n) <= 0:
                n = 100
        except ValueError:
            n = 100

        random_keys = random.sample(self.dictionary.keys(), n)
        random_records = {key: self.dictionary[key] for key in random_keys}

        df = pd.DataFrame(random_records).T
        plt.figure(figsize=(12, 26))
        sns.barplot(x=df[col], y=df.index, color='red')
        plt.title(f'Frequency of words that are {col}')
        plt.xlabel('Frequency')
        plt.ylabel('Word')
        plt.xticks(fontsize=6)
        plt.show()

    def x_word_likelihood(self, label, word, alpha):
        if word in self.dictionary:
            return (self.dictionary[word][label] + alpha) /(self.n_words_for_label[f'n_{label}'] +(len(self.dictionary) *alpha))
        else:
            return 1

    def classify(self, tokens, alpha):
        posts = { key: value for key, value in self.prior.items() }

        for word in tokens:
            for key, value in posts.items():
                posts[key] *= self.x_word_likelihood(key, word, alpha)

        # Ho hauríem de tornar com a probabilitats
        # Algo així: com esta al notebook nbc.py
        
        # posts_sum = np.sum([val for val in posts.values()])
        # for key, val in posts.items():
            # posts[key] /= posts_sum
            
        return { f'post_{key}' : value for key, value in posts.items() }

    def expandir_diccionario(self, row):
        return pd.Series(row['predicted'])

    def obtener_clave_valor_mas_alto(self, dictionary):
        all_zeros = all(value == 0 for value in dictionary.values())
        if all_zeros:
            return '??'

        max_key = max(dictionary, key=dictionary.get)
        return max_key.split('_')[1]

    def soft_eval(self, test, alpha=1):
        df = self.add_tokenize_col(test)
        df['predicted'] = df.tokens.apply(lambda x: self.classify(x, alpha))
        df_expandido = pd.concat([df, df.apply(self.expandir_diccionario, axis=1)], axis=1)
        df_expandido = df_expandido.drop('predicted', axis=1)
        return df_expandido

    def hard_eval(self, test, alpha=1):
        df = self.add_tokenize_col(test)
        df['predicted_aux'] = df.tokens.apply(lambda x: self.classify(x, alpha))
        df['predicted'] = df['predicted_aux'].apply(self.obtener_clave_valor_mas_alto)
        df = df.drop('predicted_aux', axis=1)
        return df

    def percentages_eval(self, test, alpha=1):
        df = self.add_tokenize_col(test)
        df = self.hard_eval(df, alpha)
        return pd.concat((df.groupby('Label').predicted.value_counts(), df.groupby('Label').predicted.value_counts(normalize = True)), axis = 1)