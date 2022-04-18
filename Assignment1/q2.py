import numpy as np

#A list of type float
arrList = [1,2,3,4,5,6]
newarray1 = np.array(arrList,dtype='float')
print(newarray1)

#A tuple
arrTuple=(5,10,15,20,25,30)
newarray2 = np.asarray(arrTuple)
print(newarray2)

#All zeroes
newarray3 = np.zeros(6)
print(newarray3)

#Random values
newarray4 = np.random.rand(10)
print(newarray4)

#Sequential range of integers
newarray5 = np.arange(101,110)
print(newarray5)

#Sequential range of values
newarray6 = np.arange(1,100,10)
print(newarray6)

#Reshape an array
arr = np.array([1,2,3,4,5,6,7,8])
newarray7 = arr.reshape(2,4)
print(newarray7)

#Flatten an array
newarray8 = newarray7.flatten()
print(newarray8)

