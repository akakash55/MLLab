import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1
df = pd.read_csv("Salary_Data.csv")

df.dropna(axis=0)
print(df.head(5))

X = df["YearsExperience"].to_numpy()
y = df["Salary"].to_numpy()

X = X.reshape((-1, 1))
y = y.reshape((-1, 1))

#2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

#3
epoch = 10000
learning_rate = 0.01
b1 = 0.01
b0 = 0.01
meanX = np.mean(X_train)

r2_array = []
epoch_count = []


maxr2Score = -9999999
maxEpoch = 0

#4
for i in range(1, epoch + 1):
    y_pred = b1 * X_train + b0
    error = y_pred - y_train

    j = np.mean(error, axis=0)[0]
    dj = learning_rate * j
    dm = dj / meanX
    b1 = b1 - dm
    b0 = b0 - dj

    score = r2_score(y_pred, y_train)
    r2_array.append(score)
    epoch_count.append(i)

    if score > maxr2Score:
        maxr2Score = score
        maxEpoch = i


print("B1 is:", b1)
print("B0 is:", b0)

#5
y_pred = (b1 * X_test) + b0
score = r2_score(y_test, y_pred)


print("R2 Score: " + str(score))
print("Maximum R2 score is: "+ str(maxr2Score)+ " it is reached at epoch count: "+ str(maxEpoch))

#6
plt.scatter(epoch_count, r2_array, alpha=0.6, color="purple")
plt.ylim(-10, 1)
plt.xlim(0, 501)
plt.xlabel("Epoch Count")
plt.ylabel("R2 Score")
plt.title("R2 score VS epoch count")
plt.show()