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
import random

def calcNeighbourhood(xj, dataSet, epsilon):
    N = []
    for data in dataSet:
        if (pow(sum(pow((np.array(xj) - np.array(data)), 2)), 1/2) <= epsilon):
            N.append(data)
    return N

def BDSCAN(dataSet, epsilon, MinPts):
    PI = [] #核心对象
    C = [] #簇
    for xj in dataSet:
        N = calcNeighbourhood(xj, dataSet, epsilon)
        if len(N) >= MinPts:
            PI.append(xj)
    k = 0 #初始化聚类簇数
    gamma = dataSet.copy() #初始化未访问过的样本集合
    while len(PI) != 0:
        gammaOld = gamma.copy() #记录当前未访问样本集合
        o = random.choice(PI) #选择一个核心对象
        Q = [o] #初始化队列
        gamma.remove(o)
        while (len(Q) != 0):
            q = Q.pop(0) #取出队列中首个样本
            Nq = calcNeighbourhood(q, dataSet, epsilon)
            if len(Nq) >= MinPts:
                deta = []
                for item in Nq: #求交集
                    if item in gamma:
                        deta.append(item)
                        gamma.remove(item)
                Q = Q + deta #将deta的样本加入队列Q
        k = k + 1
        
        for item in gamma:
            gammaOld.remove(item)
        Ck = gammaOld
        C.append(Ck)
        for item in Ck:
            if item in PI:
                PI.remove(item)
    return C,k

mydata = creatDataSet()
c, k = BDSCAN(mydata, 0.11, 5)
print(c, k)
import matplotlib.pyplot as plt
if k == 4:
    for data in c[0]:
        plt.scatter(data[0], data[1], c = 'r')
    for data in c[1]:
        plt.scatter(data[0], data[1], c = 'b')
    for data in c[2]:
        plt.scatter(data[0], data[1], c = 'g')
    for data in c[3]:
        plt.scatter(data[0], data[1], c = 'y')
plt.show()