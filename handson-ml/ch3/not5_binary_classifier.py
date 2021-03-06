from sklearn.base import BaseEstimator
import numpy as np
from sklearn.model_selection import cross_val_score


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


never_5_clf = Never5Classifier()
score = cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

print(score)
