import numpy as np
import scipy.linalg as lina

def calcCovarianceMatrix(X, Y = None): #计算协方差矩阵
    m = X.shape[0]
    X = X - np.mean(X, axis = 0)
    if Y == None:
        Y = X
    else:
        Y = np.mean(Y, axis = 0)
    return 1 / m * np.matmul(X.T, Y)

def PCA(X, q): #将n维数据降为q维
    covariance_matrix = calcCovarianceMatrix(X)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix) #获取特征值，特征向量
    idx = np.argsort(eigenvalues[::-1]) #返回特征值从大到小的下标
    eigenvectors = eigenvectors[:, idx] #特征向量按照从大到小排序
    eigenvectors = eigenvectors[:, :q]
    return np.matmul(X, eigenvectors)

x = np.random.rand(10,5)


print(PCA(x,2))