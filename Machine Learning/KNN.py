import numpy as np

def creatDataSet():
    dataSet = [
        [0.697,0.46,1],
        [0.774,0.376,1],
        [0.634,0.264,1],
        [0.608,0.318,1],
        [0.556,0.215,1],
        [0.403,0.237,1],
        [0.481,0.149,1],
        [0.437,0.211,1],
        [0.666,0.091,0],
        [0.243,0.267,0],
        [0.245,0.057,0],
        [0.343,0.099,0],
        [0.639,0.161,0],
        [0.657,0.198,0],
        [0.36,0.37,0],
        [0.593,0.042,0],
        [0.719,0.103,0],
    ]
    labels = ['密度','含糖率','好瓜']
    return dataSet, labels

def calckNeighbourhood(sample, k, dataSet):
    D = []
    N = []
    for data in dataSet: #计算出样本点和数据集所有样本的距离
        d = pow(sum(pow((np.array(sample) - np.array(data[:-1])), 2)), 1/2)
        D.append(d)
    dataSetCopy = list.copy(dataSet)
    for i in range(0, k): #找出距离最近的k个样本
        min_index = np.argmin(D)
        N.append(dataSetCopy[min_index])
        D.pop(min_index)
        dataSetCopy.pop(min_index)
    return N

def kNN(sample, k, dataSet, mode): #k为k邻近，sample为样本，mode为分类（0）或者预测（1）
    N = calckNeighbourhood(sample, k, dataSet)
    if mode == 0:
        Class = {}
        for data in N: #计算k邻近的类别个数
            if data[-1] not in Class:
                Class[data[-1]] = 0
            Class[data[-1]] += 1
        maxValue = max(Class.values())
        for key,value in Class.items():
            if value == maxValue:
                return key
    elif mode == 1:
        avg = 0
        for data in N:
            avg += data[-1]
        avg = avg / k
        return avg
dataSet, labels = creatDataSet()
result = kNN([0.743,0.432], 3, dataSet, 0)
print(result)