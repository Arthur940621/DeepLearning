import math
import numpy as np

def creatDataSet():
    x = [0,   1,   2,   3,   4,   5,   6,   7,   8,   9]
    y = [1,   1,   1,   -1,  -1,  -1,  1,   1,   1,   -1]
    w = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    return x, y, w

def creatThreshs(x): #计算所有阈值的取值
    threshs = []
    for i in range(len(x) + 1):
        threshs.append(i - 1 + 0.5)
    return threshs

def allThreshResult(x, threshs): #计算所有阈值划分结果，包含正向与反向
    yAllPs = {}
    yAllNg = {}
    for thresh in threshs:
        yPs = []
        yNg = []
        for i in range(len(x)):
            if x[i] < thresh:
                yPs.append(1)
                yNg.append(-1)
            else:
                yPs.append(-1)
                yNg.append(1)
        yAllPs[thresh] = yPs
        yAllNg[thresh] = yNg
    return yAllPs, yAllNg

def calOneErrorRate(y, w, yAll, thresh): #计算一个阈值的错误率
    e = 0
    for i in range(len(y)):
        if yAll[thresh][i] != y[i]:
            e += w[i]
    return e

def calErrorRate(y, threshs, w, yALL): #计算所有阈值错误率
    es = {}
    emin = 1
    n = 0
    for thresh in threshs:
        e = calOneErrorRate(y, w, yALL, thresh)
        es[thresh] = round(e, 6)
        if e < emin:
            emin = round(e, 6)
            n = thresh
    return es, emin, n

def calAllErrorRate(x, y, threshs, w, yAllPs, yAllNg):
    esPs, eminPs, nPs = calErrorRate(y, threshs, w, yAllPs)
    esNg, eminNg, nNg = calErrorRate(y, threshs, w, yAllNg)
    if eminPs <= eminNg:
        return esPs, eminPs, nPs, yAllPs
    else:
        return esNg, eminNg, nNg, yAllNg

def BestThreshFunc(yAll, n): #获得最佳阈值对应的预测函数
    pre_n = yAll[n]
    return pre_n

def calcAlphan(emin): #计算模型权重a
    an = 0.5 * math.log((1 - emin) / emin)
    an = round(an, 6)
    return an

def calcNewW(x, y, pre_n, wn, an): #计算新的权重
    wnTemp = []
    for i in range(len(x)):
        wnew = wn[i] * math.exp(-1 * an * y[i] * pre_n[i])
        wnew = round(wnew, 6)
        wnTemp.append(wnew)
    zn = sum(wnTemp)
    wn_new = [round(i / zn, 5) for i in wnTemp]
    return wn_new

def g_n(x, pre_n, an):
    gn = []
    for i in range(len(x)):
        gnSingle = an * pre_n[i]
        gn.append(gnSingle)
    return gn

def Adaboost(x, y, w, T):
    a_n_s = [] #存放个体学习器权重
    w_n_s = [] #存放样本权重
    pre_n_s = [] #存放预测值
    g_n_s = [] #存放加权后的预测值
    n_s = []

    #权重初始化
    w_n_s.append(w)
    #生成阈值
    threshs = creatThreshs(x)
    for i in range(T):
        wn_new = w_n_s[i]
        yAllPs, yAllNg = allThreshResult(x, threshs)
        es, emin, n, yAll = calAllErrorRate(x, y, threshs, wn_new, yAllPs, yAllNg)
        pre_n = BestThreshFunc(yAll, n)
        an = calcAlphan(emin)
        wn_new = calcNewW(x, y, pre_n, wn_new, an)
        gn = g_n(x, pre_n, an)

        #数据存储
        a_n_s.append(an)
        w_n_s.append(wn_new)
        pre_n_s.append(pre_n)
        g_n_s.append(gn)
        n_s.append(n)
    
    return a_n_s, w_n_s, pre_n_s, n_s


x, y, w = creatDataSet()
a_n_s, w_n_s, pre_n_s, n_s = Adaboost(x, y, w, 3)
print(a_n_s)