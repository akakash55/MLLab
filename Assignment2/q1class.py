import numpy as np

x = np.arange(16)
x = x.reshape(4,4)

print(x)

[arr1,arr2] = np.array_split(x, 2)
print(arr1)
print(arr2)

newarr=np.vstack((arr1,arr2))
print(newarr)