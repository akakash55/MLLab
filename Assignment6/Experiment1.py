import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

#1
data = pd.read_csv("Salary_Data.csv")

X = data.iloc[:, 0]
y = data.iloc[:, 1]
print(X)
print(y)

#2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

plt.scatter(X, y)
plt.show()

#3
m = 0
c = 0
learning_rate = 0.01  
epochs = 10000  

n = float(len(X))

#4
for i in range(epochs):
    Y_pred = (m * X) + c
    D_m = (-2 / n) * sum(X * (y - Y_pred))  
    D_c = (-2 / n) * sum(y - Y_pred)  
    m = m - learning_rate * D_m  
    c = c - learning_rate * D_c  
   
print (m, c)

#5
y_pred = []
for i in X_test:
    y_pred.append(m * i + c)


print(f'r2_score: {r2_score(y_test, y_pred)}')

#6
plt.scatter(X_test, y_pred, alpha=0.6, label="Predicted")
plt.scatter(X_test, y_test, alpha=0.6, label="Actual")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Years of Experience VS Salary")
plt.legend()
plt.show()