y_train_pred_90 = y_scores > 70000

precision_90 = precision_score(y_train_5, y_train_pred_90)
recall_with_precision_90 = recall_score(y_train_5, y_train_pred_90)

print("Precision with Precision >== 90:", precision_90)
print("Recall with Precision >== 90:", recall_with_precision_90)
