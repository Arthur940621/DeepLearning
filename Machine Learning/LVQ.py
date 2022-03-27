import numpy as np
import random
def creatDataSet():
    dataSet = [
        [0.697,0.46,0],
        [0.774,0.376,0],
        [0.634,0.264,0],
        [0.608,0.318,0],
        [0.556,0.215,0],
        [0.403,0.237,0],
        [0.481,0.149,0],
        [0.437,0.211,0],
        [0.666,0.091,1],
        [0.243,0.267,1],
        [0.245,0.057,1],
        [0.343,0.099,1],
        [0.639,0.161,1],
        [0.657,0.198,1],
        [0.36,0.37,1],
        [0.593,0.042,1],
        [0.719,0.103,1],
        [0.359,0.188,1],
        [0.339,0.241,1],
        [0.282,0.257,1],
        [0.748,0.232,1],
        [0.714,0.346,0],
        [0.483,0.312,0],
        [0.478,0.437,0],
        [0.525,0.369,0],
        [0.751,0.489,0],
        [0.532,0.472,0],
        [0.473,0.376,0],
        [0.725,0.445,0],
        [0.446,0.459,0],
    ]
    return dataSet

def calcMinDistance(x, p): #计算最近的原型向量
    pNear = p[0]
    pNearIndex = 0
    dMin = pow(sum(pow((np.array(x[0:-1]) - np.array(p[0])), 2)), 1/2)
    i = -1
    for pi in p:
        i += 1
        d = pow(sum(pow((np.array(x[0:-1]) - np.array(pi)), 2)), 1/2)
        if d < dMin:
            pNear = pi
            pNearIndex = i
    return pNear, pNearIndex


def LVQ(dataSet, q, t, lamda, iteration):
    l = len(dataSet)
    p = random.sample(dataSet, q) #随机选择q个样本作为原型向量
    for i in range(0, q):
        p[i] = p[i][0:-1]

    for i in range(1, iteration):
        for xj in dataSet:
            pNear, pNearIndex = calcMinDistance(xj, p)
            if xj[-1] == t[pNearIndex]:
                pstar = np.array(pNear) + lamda * (np.array(xj[0:-1]) - np.array(pNear))
            else:
                pstar = np.array(pNear) - lamda * (np.array(xj[0:-1]) - np.array(pNear))
            p[pNearIndex] = pstar
    return p
mydata = creatDataSet()
p = LVQ(mydata, 5, [0,1,1,0,0], 0.1,1000)

import matplotlib.pyplot as plt
for data in mydata:
    if data[-1] == 0:
        plt.scatter(data[0], data[1], c = 'r')
    elif data[-1] == 1:
        plt.scatter(data[0], data[1], c = 'b')
    else:
        plt.scatter(data[0], data[1], c = 'g')
for temp in p:
    plt.scatter(temp[0], temp[1], c = 'y', marker='+')
plt.show()