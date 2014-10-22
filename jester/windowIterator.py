import numpy as np
from time import time

A = np.ones((100,4000))
B = np.ones((100,100))
t=time()
for i in range(1000):
#    a = np.ones((1000))
#    A = np.append([a],A,axis=0)[:-1,:]
    C = B.dot(A)
print time() -t

t=time()
for i in range(1000):
    a = np.ones((4000))
    A = np.append([a],A,axis=0)[:-1,:]
    C = B.dot(A)
print time() -t

A = np.zeros((1100,4000))
for i in range(100):
    A[i,:] = np.ones(4000)
t = time()
for i in range(1000):
    a = np.ones((4000))
    A[i+100,:] = a
    C = B.dot(A[i+1:(i+100+1),
print time()-t
