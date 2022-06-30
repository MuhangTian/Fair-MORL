from cmath import e
import numpy as np

arr = np.zeros([1, 6, 6])
arr2 = np.array([1,2,3,4,5,6])
sum = arr + arr2
print(sum[0, 1])
print(np.log(e))