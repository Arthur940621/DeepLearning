import numpy as np

X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
Y = np.array([[0,1,1,0]]).T
w1 = 2 * np.random.random((3, 4)) - 1
w2 = 2 * np.random.random((4, 1)) - 1

for it in range(10000):
    #前向传播
    a0 = X
    z1 = np.exp(-np.dot(a0, w1))
    a1 = 1 / (1 + z1)
    z2 = np.exp(-np.dot(a1, w2))
    a2 = 1 / (1 + z2)
    #反向传播
    delta2 = (a2 - Y) * (a2 * (1 - a2))
    delta1 = np.dot(w2.T, delta2) * (a1 * (1 - a1))
    #更新权重
    w2 -= np.dot(a1.T, delta2)
    w1 -= np.dot(a0.T, delta1)

print(w1,w2)