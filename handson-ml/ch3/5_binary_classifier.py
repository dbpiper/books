from sklearn.linear_model import SGDClassifier

# target vectors which only contain '5's
y_train_5 = y_train == 5
y_test_5 = y_test == 5

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

print(sgd_clf.predict([some_digit]))
