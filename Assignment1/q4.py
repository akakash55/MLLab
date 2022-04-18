import numpy as np

arr1 = np.array([[10,20,30,40],[20,40,60,80]])
arr2 = np.array([[20,40,60,80],[10,20,30,40]])

print(arr1)
print(arr2)

sumarr = arr1+arr2
multiplicationarr = arr1*arr2

print("Sum is: ",sumarr)
print("Multiplication is: ",multiplicationarr)
