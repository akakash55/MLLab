import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

table = pd.read_csv("brain.csv")

table.dropna(axis = 0)
table.head()

Y=table['Brain Weight(grams)']
print(Y)
table.drop('Brain Weight(grams)',axis=1,inplace=True)
X=table
print(X)

X = X.to_numpy()
Y = Y.to_numpy()
print(X.shape,Y.shape)

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state=12)
print(x_train.shape,x_test.shape)