from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


class ModelBuilding:
    model = None

    def __init__(self):
        return

    def logisticRegression(self, X_train, y_train, X_test, y_test):
        self.model = LogisticRegression(solver='liblinear')
        self.model.fit(X_train, y_train)
        print('Score on training data is: ' + str(self.model.score(X_train, y_train)))
        print('Score on testing data is: ' + str(self.model.score(X_test, y_test)))
        joblib.dump(self.model, 'model.pkl')
