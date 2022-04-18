import py_compile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('diabetes.csv')


y=df['Outcome'].to_numpy()
df.drop('Outcome',axis=1,inplace=True)
x=df.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)

clf=SVC()

clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix = \n" + str(matrix))


print("Accuracy = ",accuracy_score(y_test,y_pred))
print("Precision = ",precision_score(y_test,y_pred))
print("Recall = ",recall_score(y_test,y_pred))
print("F1-Score = ",f1_score(y_test,y_pred))