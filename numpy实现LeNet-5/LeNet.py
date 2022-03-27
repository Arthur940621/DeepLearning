import numpy as np
from os import listdir
import matplotlib.pyplot as plt
#将图像转换为向量
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

#读取数据
def handwritingData(dataPath):
    hwLabels = []
    fileList = listdir(dataPath) #获取目录内容
    m = len(fileList)
    np.Mat = np.zeros((m, 1024))
    #从文件名解析分类数字
    for i in range(m):
        fileNameStr = fileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        np.Mat[i, :] = img2vector(dataPath + '/%s' % fileNameStr)
    return np.Mat, hwLabels


def Sigmoid(x, diff = False):
    def sigmoid(x): #sigmoid函数
        return 1 / (1 + np.exp(-x))
    def dsigmoid(x): #sigmoid函数求导
        f = sigmoid(x)
        return f * (1 - f)
    if diff == True:
        return dsigmoid(x)
    return sigmoid(x)

#平方损失
def SquareErrorSum(y_hat, y, diff = False):
    if diff == True:
        return y_hat - y
    return(np.square(y_hat - y) * 0.5).sum()


class Net():
    def __init__(self):
        self.X =  np.random.randn(1024, 1) #输入层

        self.W1 = np.random.randn(16, 1024) #L1层权重
        self.b1 = np.random.randn(16, 1) #L1层偏置

        self.W2 = np.random.randn(16, 16) #L2层权重
        self.b2 = np.random.randn(16, 1) #L2层偏置

        self.W3 = np.random.randn(10, 16) #输出层权重
        self.b3 = np.random.randn(10, 1) #输出层偏置

        self.alpha = 0.01 #学习率
        self.losslist = [] #损失
    
    def forward(self, X, y, activate): #前向传播
        self.X = X

        self.z1 = np.dot(self.W1, self.X) + self.b1 #16*1024 * 1024*1 + 16*1 = 16*1
        self.a1 = activate(self.z1) #激活函数
        
        self.z2 = np.dot(self.W2, self.a1) + self.b2 #16*16 * 16*1 + 16*1 = 16*1
        self.a2 = activate(self.z2)

        self.z3 = np.dot(self.W3, self.a2) + self.b3 #10*16 * 16*1 + 10*1 = 10*1
        self.y_hat = activate(self.z3)
        Loss = SquareErrorSum(self.y_hat, y)
        return Loss, self.y_hat

    def backward(self, y, activate): #后向传播
        self.delta3 = activate(self.z3, True) * SquareErrorSum(self.y_hat, y, True)
        self.delta2 = activate(self.z2, True) * (np.dot(self.W3.T, self.delta3))
        self.delta1 = activate(self.z1, True) * (np.dot(self.W2.T, self.delta2))

        dW3 = np.dot(self.delta3, self.a2.T)
        dW2 = np.dot(self.delta2, self.a1.T)
        dW1 = np.dot(self.delta1, self.X.T)

        d3 = self.delta3
        d2 = self.delta2
        d1 = self.delta1

        #更新权重
        self.W3 -= self.alpha * dW3
        self.W2 -= self.alpha * dW2
        self.W1 -= self.alpha * dW1
        self.b3 -= self.alpha * d3
        self.b2 -= self.alpha * d2
        self.b1 -= self.alpha * d1

    def setLearnrate(self, l): #设定学习率
        self.alpha = l

    def train(self, trainMat, trainLabels, Epoch = 5):
        for epoch in range(Epoch):
            acc = 0.0
            acc_cnt = 0
            label = np.zeros((10, 1)) #生成一个10x1是向量，减少运算
            for i in range(len(trainMat)):
                X = trainMat[i, :].reshape((1024, 1)) #X是一个1024*1的列向量作为输入

                labelidx = trainLabels[i]
                label[labelidx][0] = 1.0

                Loss, y_hat = self.forward(X, label, Sigmoid) #前向传播
                self.backward(label, Sigmoid) #反向传播

                label[labelidx][0] = 0.0 #还原为0向量
                acc_cnt += int(trainLabels[i] == np.argmax(y_hat)) #如果标签不一致，个数加1

            acc = acc_cnt / len(trainMat) #计算每一次迭代的精确度
            self.losslist.append(Loss) #保存每一次的损失值
            if(epoch == 0 or epoch == 99 or epoch == 499 or epoch == 999 or epoch == 1499 or epoch == 1999):
                print("epoch:%d,loss:%02f,accrucy : %02f" % (epoch, Loss, acc))
        self.plotLosslist(self.losslist, "Loss")

    def plotLosslist(self, Loss, title): #绘图
        font = {'family': 'simsun',
                'weight': 'bold',
                'size': 20,
                }
        m = len(Loss)
        X = range(m)
        # plt.figure(1)
        plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
        plt.subplot(111)
        plt.title(title, font)
        plt.plot(X, Loss)
        plt.xlabel(r'Epoch', font)
        plt.ylabel(u'Loss', font)
        plt.show()

    def test(self, testMat, testLabels): #测试
        acc = 0.0
        acc_cnt = 0
        label = np.zeros((10, 1))
        m = len(testMat)
        for i in range(m):
            X = testMat[i, :].reshape((1024, 1))

            labelidx = testLabels[i]
            label[labelidx][0] = 1.0

            Loss, y_hat = self.forward(X, label, Sigmoid)

            label[labelidx][0] = 0.0
            acc_cnt += int(testLabels[i] == np.argmax(y_hat))
        acc = acc_cnt / m
        print("test num: %d, accrucy : %05.3f%%"%(m, acc*100))

#读取训练数据
trainDataPath = "C:\\Users\\Arthur\\Desktop\\代码\\LeNet\\trainingDigits"
trainMat, trainLabels = handwritingData(trainDataPath)
#读取测试数据
testDataPath = "C:\\Users\\Arthur\\Desktop\\代码\\LeNet\\testDigits"
testMat, testLabels = handwritingData(testDataPath)

net = Net()
net.setLearnrate(0.01) #设定学习率

net.train(trainMat, trainLabels, Epoch = 2000) #训练
net.test(testMat, testLabels) #测试
