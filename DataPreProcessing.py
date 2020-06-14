import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from ModelBuilding import ModelBuilding
from sklearn.externals import joblib


class DataPreProcessing:

    cleaned_data = None
    vectorizer = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    def __init__(self):
        self.cleaned_data = pd.read_csv('cleaned_reviews.csv')
        self.vectorizer = TfidfVectorizer(sublinear_tf=True,
                                          encoding='utf-8',
                                          decode_error='ignore')

    def trainTestSplit(self):
        self.X_train = self.cleaned_data.loc[:35000, 'review'].values
        self.y_train = self.cleaned_data.loc[:35000, 'sentiment'].values
        self.X_test = self.cleaned_data.loc[35000:, 'review'].values
        self.y_test = self.cleaned_data.loc[35000:, 'sentiment'].values
        return (self.X_train, self.y_train, self.X_test, self.y_test)

    def tfidf(self):
        self.vectorizer.fit(self.X_train)
        X_train_tfid = self.vectorizer.transform(self.X_train)
        X_test_tfid = self.vectorizer.transform(self.X_test)
        joblib.dump(self.vectorizer, 'vectorizer.pkl')
        m = ModelBuilding()
        m.logisticRegression(X_train_tfid, y_train, X_test_tfid, y_test)


if __name__ == '__main__':
    d = DataPreProcessing()
    (X_train, y_train, X_test, y_test) = d.trainTestSplit()
    d.tfidf()
