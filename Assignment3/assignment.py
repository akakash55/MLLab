import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df=pd.read_csv('fruit-production1.csv')
df.set_index('Year', inplace=True)

df.plot(kind='line')
plt.xlabel("Years")
plt.ylabel("Fruits")
plt.show()
