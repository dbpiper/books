from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
y_train_confusion_matrix = confusion_matrix(y_train_5, y_train_pred)

print(y_train_confusion_matrix)
