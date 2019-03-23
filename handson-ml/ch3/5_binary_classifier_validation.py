from sklearn.model_selection import cross_val_score

score = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

print(score)
