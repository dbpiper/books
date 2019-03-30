sgd_clf.fit(X_train, y_train)
prediction = sgd_clf.predict([some_digit])

print("Predicted digit:", prediction)
