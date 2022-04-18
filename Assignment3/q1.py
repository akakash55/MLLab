import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df=pd.read_csv('food-grains.csv')

print(df.columns)
print(df.shape)
print(df.info())
print(df.describe())
print(df.head(15))
print(df.tail(10))

rdf=df[df['State'] =="Jharkhand"]
rdf=rdf.drop('State',axis=1) #drops the state in rdf
print("Resultant row:\n", rdf)

# df.plot(kind='bar',x='State')
# plt.show()

# rdf.plot(kind='bar')
# plt.show()

df.set_index('State', inplace=True) #Row ka name 0,1,2 se states ho jayega
print(df.columns)
print(df.info())
maxValues = df.max(axis = 1)
print("State wise max.\n")
print(maxValues)

max_each_state = df.max(axis=0)
print("Food wise max.\n")
print(max_each_state)



for i,col in enumerate(df.columns):
	max_amt=df[col].max()
	state=df[df[col]==max_amt].iloc[0].name
	print(f"Highest producer of {col} is {state},producing {max_amt}.")