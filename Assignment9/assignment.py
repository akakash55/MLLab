from pyexpat.errors import XML_ERROR_TAG_MISMATCH
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris

df = pd.read_csv('diabetes.csv')

Y = df['Outcome']
df.drop('Outcome',axis=1,inplace=True)
X = df
print(Y.shape)
print(Y)
print(X.shape)
print(X)

Y = Y.to_numpy()
X = X.to_numpy()
print(Y.shape)
print(Y)
print(X.shape)
print(X)

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state=12)

scaler = StandardScaler()

scaler.fit(x_train)
x_train_std = scaler.transform(x_train)
print(x_train_std)
x_test_std = scaler.transform(x_test)

clf1 = LogisticRegression(C=100.0, max_iter=10000)

clf1.fit(x_train_std,y_train)

y_train_pred = clf1.predict(x_train_std)

print('Accuracy Score on training data: ',accuracy_score(y_train,y_train_pred))
print(confusion_matrix(y_train,y_train_pred))

y_test_pred = clf1.predict(x_test_std)

print('Accuracy Score on testing data: ',accuracy_score(y_test,y_test_pred))
print(confusion_matrix(y_test,y_test_pred))



#Problem 2


iris = load_iris()

x = iris.data
y = iris.target
print(x.shape,y.shape)

xtrain, xtest, ytrain,ytest = train_test_split(x,y,test_size=0.2, random_state=42)
print(xtrain.shape,xtest.shape)

scaler.fit(xtrain)
xtrain_std = scaler.transform(xtrain)
xtest_std = scaler.transform(xtest)

clf2 = LogisticRegression(multi_class="ovr")

clf2.fit(xtrain_std,ytrain)

ytrain_pred = clf2.predict(xtrain_std)

print('Accuracy Score on training data: ',accuracy_score(ytrain,ytrain_pred))
print(confusion_matrix(ytrain,ytrain_pred))

ytest_pred = clf2.predict(xtest_std)

print('Accuracy Score on testing data: ',accuracy_score(ytest,ytest_pred))
print(confusion_matrix(ytest,ytest_pred))