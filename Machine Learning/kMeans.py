import numpy as np
import pandas as pd
import math
import random
def creatDataSet():
    dataSet = [
        [0.697,0.460],
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
        [0.360,0.370],
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
    labels = ['密度','含糖率']
    return dataSet, labels
def calcMinDistance(x, u): #计算最近的簇
    d = []
    for i in range(0, len(u)):
        d.append(pow(sum(pow((x - u[i]), 2)), 1/2))
    d = np.array(d)
    c_j = np.where(d == d.min())
    return c_j[0][0]

def K_Mean(dataSet, k, iteration):
    dataSet = np.array(dataSet)
    u = [] #均值向量
    c = [] #簇
    for i in range(0, k):
        c.append([])
    l = len(dataSet)

    r = random.sample(range(0, l), k)
    for i in r: #随机选择k个样本作为初始均值向量
        u.append(np.array(dataSet[i]))

    for i in range(0, iteration):
        c = [] #簇
        for i in range(0, k):
            c.append([])
        for i in range(0, l):
            c_j = calcMinDistance(dataSet[i], u)
            c[c_j].append(dataSet[i])
        u = []
        for i in range(0, k):
            Sum = sum(np.array(c[i]))
            Len = len((np.array(c[i])))
            u.append((Sum / Len))
    return c, u
       

mydata, label = creatDataSet()
c, u = K_Mean(mydata, 3, 4)

import matplotlib.pyplot as plt
for data in c[0]:
    plt.scatter(data[0], data[1], c = 'r')
for data in c[1]:
    plt.scatter(data[0], data[1], c = 'b')
for data in c[2]:
    plt.scatter(data[0], data[1], c = 'g')

for temp in u:
    plt.scatter(temp[0], temp[1], c = 'y', marker='+')

plt.show()