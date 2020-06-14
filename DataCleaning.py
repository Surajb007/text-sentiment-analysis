import urllib.request
import os
import pandas as pd
import numpy as np
# import nltk
# nltk.download()
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


class DataCleaning:

    df = None
    tokenizer = None
    en_stopwords = None
    ps = None
    path = './New folder/aclImdb_v1'

    def __init__(self):
        # init Objects
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.en_stopwords = set(stopwords.words('english'))
        self.ps = PorterStemmer()
        print('Objects Created')
        self.df = pd.read_csv('movie_data.csv', encoding='utf-8')

    def getStemmedReview(self, review):
        review = review.lower() # Lower Case
        review = review.replace("<br /><br />", " ") # Removing new lines
        tokens = self.tokenizer.tokenize(review) # Tokenize
        new_tokens = [token for token in tokens if token not in self.en_stopwords] # Removing stop words
        stemmed_tokens = [self.ps.stem(token) for token in new_tokens] # Stemming
        clean_review = ' '.join(stemmed_tokens) # Joining the cleaned sentence back
        return clean_review

    def trainTestSplit(self, df):
        X_train = df.loc[:35000, 'review'].values
        y_train = df.loc[:35000, 'sentiment'].values
        X_test = df.loc[35000:, 'review'].values
        y_test = df.loc[35000:, 'sentiment'].values
        return (X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    d = DataCleaning()
    # d.df['review'].apply(d.getStemmedReview)
    # d.df.to_csv('cleaned_reviews.csv')
    df = pd.read_csv('cleaned_reviews.csv')
    (X_train, y_train, X_test, y_test) = d.trainTestSplit(df)
