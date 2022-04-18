import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#1
table = pd.read_csv("Salary_Data.csv")

#2
table.dropna(axis = 0)
table.head()

#3
X = table['YearsExperience'].to_numpy()
y = table['Salary'].to_numpy()
print(X)
print(y)
print()

#4
X = X.reshape((30, 1))
y = y.reshape((30, 1))
print(X)
print(y)
print()

#5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
print(X_train)
print()

#6
n = len(X_train)
X_mean = np.mean(X_train, axis = 0)[0]
y_mean = np.mean(y_train, axis = 0)[0]
print(X_mean)
X_dev = X_train - X_mean
y_dev = y_train - y_mean
print(X_dev)
covariance = 0
variance = 0
for i in range(n):
    covariance = covariance + X_dev[i][0]*y_dev[i][0]
    variance = variance + X_dev[i][0]*X_dev[i][0]
variance = variance / n
covariance = covariance / n
print("X mean: " + str(X_mean))
print("y_mean: " + str(y_mean))
print("Variance of X_train: " + str(variance))
print("Co-variance of X_train with y_train: " + str(covariance))
print()

#7
B1 = covariance / variance
B0 = y_mean - B1*X_mean
print("Value of B1: " + str(B1))
print("Value of B0: " + str(B0))
print("Equation for Regression line: (y = " + str(B1) + " * x + " + str(B0) + ")")
print()

#8
m = len(X_test)
y_predict = np.zeros((m, 1))
for i in range(m):
    y_predict[i][0] = X_test[i][0]*B1 + B0
print()

#9
score = r2_score(y_test, y_predict)
print("R2 Score: " + str(score))
print()

#10
plt.scatter(X_test, y_predict, alpha=0.6)
plt.scatter(X_test, y_test, alpha=0.6)
plt.legend(["Predicted", "Actual"])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()