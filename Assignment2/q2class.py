import numpy as np
import pandas as pd

#a=pd.Series({'Jan':100,'Feb':200,'March':400,'April':50})
#print(a)

#b=pd.Series({'Fee':300,'Address':'Ranchi'})
#print(b)
#print(b.dtype)

c=pd.Series({'Games':10,'Creative Art':20,'NSS':30,'NCC':40})
d=pd.Series({'Games':40,'Creative Art':30,'NSS':20,'NCC':10})

print("Total number of student in games stream = ",c['Games']+d['Games'])
print("Total number of student in creative art stream = ",c['Creative Art']+d['Creative Art'])
print("Total number of student in nss stream = ",c['NSS']+d['NSS'])
print("Total number of student in ncc stream = ",c['NCC']+d['NCC'])

total=c.add(d)
print(total)