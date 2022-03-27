from math import log
import operator

def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],               
               [0, 0, 0, 0, 'no'],               
               [1, 0, 0, 0, 'no'],               
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],               
               [1, 0, 1, 2, 'yes'],               
               [1, 0, 1, 2, 'yes'],               
               [2, 0, 1, 2, 'yes'],               
               [2, 0, 1, 1, 'yes'],               
               [2, 1, 0, 1, 'yes'],               
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['age', 'work', 'house', 'credit', 'apply']
    return dataSet, labels

#计算经验熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    laberCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in laberCounts.keys():
            laberCounts[currentLabel] = 0
        laberCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in laberCounts:
        prob = float(laberCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

#按照给定的特征划分数据集，axis是特征下标，value是匹配的值，例如axis取0（年龄）时，value可以取0（青年），1（中年），2（老年）
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#计算某个特征的信息增益
def calInfoGain(dataSet, axis):
    baseEntropy = calcShannonEnt(dataSet)
    newEntropy = 0.0
    featList = [example[axis] for example in dataSet]
    uniqueVals = set(featList)
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet, axis, value)
        prob = len(subDataSet) / float(len(dataSet))
        newEntropy += prob * calcShannonEnt(subDataSet)
    infoGain = baseEntropy - newEntropy
    return infoGain

#ID3选择最佳特征
def chooseBestFeatureToSplitID3(dataSet):
    numFeatures = len(dataSet[0]) - 1
    BaseEntropy = calcShannonEnt(dataSet) #计算原始数据集的熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures): #遍历所有特征
        featList = [example[i] for example in dataSet] #featList是每一列的所有值，是一个列表
        uniqueVals = set(featList) #集合中每个值互不相同
        newEntropy = 0.0
        for value in uniqueVals: #扫描所有特征的取值
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = BaseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature #返回最佳特征的下标

def majorityCnt(classList): #多数投票，如果特征都用完了，还没有划分好，用该函数选择所属的类
    classCount = {} #列标签字典，存储每类标签出现的次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True) #降序排序的字典
    return sortedClassCount[0][0] #返回出现次数最多的分类名称

def createTreeID3(dataSet, labels):
    classList = [example[-1] for example in dataSet] #创建类标签列表
    if classList.count(classList[0]) == len(classList): #中止条件1：如果所有类标签完全相同则停止
        return classList[0]
    if len(dataSet[0]) == 1: #中止条件2：遍历完所有标签仍然不能将数据集划分为仅包含唯一类别的分组，因为每次划分会减少特征，划分到最后只有类别变量，所以为1
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplitID3(dataSet) #最佳分类特征的下标
    bestFeatLabel = labels[bestFeat] #最佳分类的名称
    myTree = {bestFeatLabel:{}} #创建字典，存储树的所有信息
    del (labels[bestFeat]) #从特征里删除掉这个最佳特征
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals: #遍历最佳特征所有特征值，继续递归划分子树
        subLabels = labels[:] #拷贝，防止构建子树时删除特征相互干扰
        myTree[bestFeatLabel][value] = createTreeID3(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

mydata, labels = createDataSet()
#print(createTreeID3(mydata, labels))

import random
def RandomForest(dataSet, labels, m, T):
    Trees = []
    sample = []
    mlabels = []
    mlabelsindex = []
    N = len(dataSet)
    for i in range(N): #随机挑选N个样本
        r = random.randint(0, N - 1)
        sample.append(dataSet[r])
    mlabelsindex = random.sample(range(0, len(labels) - 1), m) #随机生成m个特征的下标
    for i in mlabelsindex: #提取出m个特征值
        mlabels.append(labels[i])
    new_sample = []
    
    for i in range(N):
        s = sample[i]
        temp = []
        for index in mlabelsindex:
            temp.append(s[index])
        temp.append(s[-1])
        new_sample.append(temp)
    print(new_sample)
RandomForest(mydata, labels, 2, 4)