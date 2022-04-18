import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



iris_data = load_iris()
X = iris_data['data']
y = iris_data['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svc = SVC()
svc.fit(X_train, y_train)


y_pred = svc.predict(X_test)
matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: " + str(matrix))


print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}')
print("Precision Score: ", precision_score(y_test, y_pred, average='weighted'))
print("Precision Score: ", precision_score(y_test, y_pred, average='weighted'))
print("Recall Score: ", recall_score(y_test, y_pred, average='weighted'))