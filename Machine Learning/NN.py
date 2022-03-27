import numpy as np

X = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
Y = np.array([[0,1,1,0]]).T
w = 2 * np.random.random((3, 1)) - 1

for it in range(10000):
    z = np.dot(X, w)
    y = 1 / (1 + np.exp(-z))
    error = y - Y
    slope = y * (1 - y)
    delta = error * slope
    w -= np.dot(X.T, delta)
print(w)