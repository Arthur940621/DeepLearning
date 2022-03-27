def creatDataSet():
    dataSet = [
        [0.697,0.46],
        [0.774,0.376],
        [0.634,0.264],
        [0.608,0.318],
        [0.556,0.215],
        [0.403,0.237],
        [0.481,0.149],
        [0.437,0.211],
        [0.666,0.091],
        [0.243,0.267],
        [0.245,0.057],
        [0.343,0.099],
        [0.639,0.161],
        [0.657,0.198],
        [0.36,0.37],
        [0.593,0.042],
        [0.719,0.103],
        [0.359,0.188],
        [0.339,0.241],
        [0.282,0.257],
        [0.748,0.232],
        [0.714,0.346],
        [0.483,0.312],
        [0.478,0.437],
        [0.525,0.369],
        [0.751,0.489],
        [0.532,0.472],
        [0.473,0.376],
        [0.725,0.445],
        [0.446,0.459],
    ]
    return dataSet

import numpy as np

def calcDistance(Ci, Cj, Type):
    d = []
    for x in Ci:
        for z in Cj:
            d.append(pow(sum(pow((np.array(x) - np.array(z)), 2)), 1/2))
    dmin = min(d)
    dmax = max(d)
    davg = sum(np.array(d)) / (len(Ci) + len(Cj))
    if Type == 'min':
        return dmin
    elif Type == 'max':
        return dmax
    elif Type == 'avg':
        return davg

def findMinClass(M, w, h):
    minD = M[0][1]
    for i in range(0, w):
        for j in range(0, h):
            if i != j and M[i][j] <= minD:
                minD = M[i][j]
                mini = i
                minj = j

    return mini, minj

def AGNES(dataSet, k, Type = 'min'):
    C = []
    m = len(dataSet)
    M = [[0 for i in range(m)] for i in range(m)] #定义一个m*m大小的二维矩阵存储簇i,j之间的距离
    for data in dataSet:
        Cj = [data]
        C.append(Cj)
    for i in range(0, m):
        for j in range(0, m):
            M[i][j] = calcDistance(C[i], C[j], Type)
            M[j][i] = M[i][j]
    q = m #设置当前聚类簇个数
    while q > k:
        mini, minj = findMinClass(M, m, m)
        C[mini] = C[mini] + C[minj]
        C.remove(C[minj])
        m = m - 1
        M = [[0 for i in range(m)] for i in range(m)]
        for i in range(0, m):
            for j in range(0, m):
                M[i][j] = calcDistance(C[i], C[j], Type)
                M[j][i] = M[i][j]
        q = q - 1
    return C

mydata = creatDataSet()
C = AGNES(mydata, 4, 'max')

import matplotlib.pyplot as plt
for i in C[0]:
    plt.scatter(i[0], i[1], c = 'r')
for i in C[1]:
    plt.scatter(i[0], i[1], c = 'g')
for i in C[2]:
    plt.scatter(i[0], i[1], c = 'b')
for i in C[3]:
    plt.scatter(i[0], i[1], c = 'y')
plt.show()