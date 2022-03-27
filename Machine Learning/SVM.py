import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

data_dict = {-1:np.array([[1,8],[2,3],[3,6]]), 1:np.array([[1,-2],[3,-4],[3,0]])}

def fit(dataSet):
    opt_dict = {}

    #从x的正半轴到负半轴180°，按照设定的步长构造transform矩阵
    rotMatrix = lambda theta: np.array([np.sin(theta), np.cos(theta)]) #y = tanθ + b
    thetastep = np.pi / 10 #步长
    transforms = [np.array(rotMatrix(theta)) for theta in np.arange(0, np.pi, thetastep)]
    
    #将数据集拉平装到一个list当中，方便处理
    all_data = []
    for yi in dataSet:
        for featureSet in dataSet[yi]:
            for feature in featureSet:
                all_data.append(feature)

    #找到最大值
    max_feature_value = max(all_data)

    #定义步长
    step_size = [max_feature_value * 0.1, max_feature_value * 0.01, max_feature_value * 0.001]

    #寻找b
    b_rang_multiple = 5
    b_multiple = 5
    latest_optimum = max_feature_value * 10

    for step in step_size: #对步长从大到小循环
        w = np.array([latest_optimum, latest_optimum])
        optimized = False

        while not optimized:
            for b in np.arange(-1 * (max_feature_value * b_rang_multiple),
                                max_feature_value, step * b_rang_multiple,
                                step * b_multiple): #对b的循环
                for transformation in transforms: #对w的循环
                    w_t = w * transformation
                    found_option = True

                    for i in data_dict: #对每个类别循环
                        for xi in data_dict[i]: #对该类别下的每个数据集循环
                            yi = i
                            if not yi * (np.dot(w_t, xi) + b) >= 1: #不满足约束条件
                                found_option = False
                                break
                        if not found_option:
                            break
                    
                    if found_option:
                        opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                
            if w[0] < 0:
                optimized = True
            else:
                w = w - step
        norms = sorted([n for n in opt_dict])
        opt_choice = opt_dict[norms[0]]
        w = opt_choice[0]
        b = opt_choice[1]
        latest_optimum = opt_choice[0][0] * step * 2
        
        return w, b
#def predit(features):

w, b = fit(data_dict)
