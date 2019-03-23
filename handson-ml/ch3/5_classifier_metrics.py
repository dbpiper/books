from sklearn.metrics import precision_score, recall_score, f1_score

precision_score_5 = precision_score(y_train_5, y_train_pred)
recall_score_5 = recall_score(y_train_5, y_train_pred)

print("Precision Score of 5 Classifier:", precision_score_5)
print("Recall Score of 5 Classifier:", recall_score_5)

f1_score_5 = f1_score = f1_score(y_train_5, y_train_pred)

print("F1 Score of 5 Classifier:", f1_score_5)
