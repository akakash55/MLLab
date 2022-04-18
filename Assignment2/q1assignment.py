import numpy as np
import pandas as pd

dept=['CSE','IT','EEE','MECH']
series1=pd.Series([35,15,30,40],index=dept)
series2=pd.Series([500000,300000,350000,250000],index=dept)

print("Creating dataframe object from the dictionary")
dataframe=dict({"Employee Count": series1, "Total Salary": series2})

data=pd.DataFrame.from_dict(dataframe)
print(data, '\n')

print("Adding a row to the dataframe")
data.at['ECE']=[50,450000]
print(data, '\n')

print("Adding Average Salary column to the dataframe")
data["Average Salary"]=data['Total Salary']/data['Employee Count']
print(data, '\n')