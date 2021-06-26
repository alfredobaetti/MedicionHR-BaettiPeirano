from LE import LE
import numpy as np

#X = np.array([[1528, 1507, 1583], [1530, 1511, 1576], [1893, 1889, 1906]]) # nxd
X = np.array([[2,3,4], [1,2,3], [5,1,6]]) # nxd

le = LE(X, dim = 1, k = 1, graph = 'k-nearest', weights = 'heat kernel', 
        sigma = 5, laplacian = 'symmetrized')
Y = le.transform()
print(X)
print(Y)