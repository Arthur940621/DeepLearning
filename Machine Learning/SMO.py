import numpy as np
import math
import random
import matplotlib.pyplot as plt

def creatDataSet():
    dataset = []
    labels = []
    DATA = [
        [3.542485, 1.977398, -1],
        [3.018896, 2.556416, -1],
        [7.55151, -1.58003, 1],
        [2.114999, -0.004466, -1],
        [8.127113, 1.274372, 1],
        [7.108772, -0.986906, 1],
        [8.610639, 2.046708, 1],
        [2.326297, 0.265213, -1],
        [3.634009, 1.730537, -1],
        [0.341367, -0.894998, -1],
        [3.125951, 0.293251, -1],
        [2.123252, -0.783563, -1],
        [0.887835, -2.797792, -1],
        [7.139979, -2.329896, 1],
        [1.696414, -1.212496, -1],
        [8.117032, 0.623493, 1],
        [8.497162, -0.266649, 1],
        [4.658191, 3.507396, -1],
        [8.197181, 1.545132, 1],
        [1.208047, 0.2131, -1],
        [1.928486, -0.32187, -1],
        [2.175808, -0.014527, -1],
        [7.886608, 0.461755, 1],
        [3.223038, -0.552392, -1],
        [3.628502, 2.190585, -1],
        [7.40786, -0.121961, 1],
        [7.286357, 0.251077, 1],
        [2.301095, -0.533988, -1],
        [-0.232542, -0.54769, -1],
        [3.457096, -0.082216, -1],
        [3.023938, -0.057392, -1],
        [8.015003, 0.885325, 1],
        [8.991748, 0.923154, 1],
        [7.916831, -1.781735, 1],
        [7.616862, -0.217958, 1],
        [2.450939, 0.744967, -1],
        [7.270337, -2.507834, 1],
        [1.749721, -0.961902, -1],
        [1.803111, -0.176349, -1],
        [8.804461, 3.044301, 1],
        [1.231257, -0.568573, -1],
        [2.074915, 1.41055, -1],
        [-0.743036, -1.736103, -1],
        [3.536555, 3.96496, -1],
        [8.410143, 0.025606, 1],
        [7.382988, -0.478764, 1],
        [6.960661, -0.245353, 1],
        [8.23446, 0.701868, 1],
        [8.168618, -0.903835, 1],
        [1.534187, -0.622492, -1],
        [9.229518, 2.066088, 1],
        [7.886242, 0.191813, 1],
        [2.893743, -1.643468, -1],
        [1.870457, -1.04042, -1],
        [5.286862, -2.358286, 1],
        [6.080573, 0.418886, 1],
        [2.544314, 1.714165, -1],
        [6.016004, -3.753712, 1],
        [0.92631, -0.564359, -1],
        [0.870296, -0.109952, -1],
        [2.369345, 1.375695, -1],
        [1.363782, -0.254082, -1],
        [7.27946, -0.189572, 1],
        [1.896005, 0.51508, -1],
        [8.102154, -0.603875, 1],
        [2.529893, 0.662657, -1],
        [1.963874, -0.365233, -1],
        [8.132048, 0.785914, 1],
        [8.245938, 0.372366, 1],
        [6.543888, 0.433164, 1],
        [-0.236713, -5.766721, -1],
        [8.112593, 0.295839, 1],
        [9.803425, 1.495167, 1],
        [1.497407, -0.552916, -1],
        [1.336267, -1.632889, -1],
        [9.205805, -0.58648, 1],
        [1.966279, -1.840439, -1],
        [8.398012, 1.584918, 1],
        [7.239953, -1.764292, 1],
        [7.556201, 0.241185, 1],
        [9.015509, 0.345019, 1],
        [8.266085, -0.230977, 1],
        [8.54562, 2.788799, 1],
        [9.295969, 1.346332, 1],
        [2.404234, 0.570278, -1],
        [2.037772, 0.021919, -1],
        [1.727631, -0.453143, -1],
        [1.979395, -0.050773, -1],
        [8.092288, -1.372433, 1],
        [1.667645, 0.239204, -1],
        [9.854303, 1.365116, 1],
        [7.921057, -1.327587, 1],
        [8.500757, 1.492372, 1],
        [1.339746, -0.291183, -1],
        [3.107511, 0.758367, -1],
        [2.609525, 0.902979, -1],
        [3.263585, 1.367898, -1],
        [2.912122, -0.202359, -1],
        [1.731786, 0.589096, -1],
        [2.387003, 1.573131, -1],
    ]
    for data in DATA:
        dataset.append(data[:-1])
        labels.append(data[-1])
    return dataset, labels



def GaussianKernelFunction(X1, X2, sigma = 1): #高斯核
    X1 = np.array(X1)
    X2 = np.array(X2)
    k = math.exp(-(np.linalg.norm(X1 - X2) ** 2) / (2 * sigma ** 2))
    return k

def PolynomialKernelFunction(X1, X2, d = 1):
    k = math.pow((np.dot(X1.T, X2) + 1), d)
    return k

def clip(alpha, L, H):  # 剪枝
    if alpha < L:
        return L
    elif alpha > H:
        return H
    else:
        return alpha


def select_j(i, m):  # 在m中随机选择除了i之外剩余的数
    l = list(range(m))
    seq = l[: i] + l[i+1:]
    return random.choice(seq)


def get_w(alphas, dataset, labels):  # 通过已知数据点和拉格朗日乘子获得分割超平面参数w
    alphas, dataset, labels = np.array(alphas), np.array(dataset), np.array(labels)
    yx = labels.reshape(1, -1).T * np.array([1, 1]) * dataset
    w = np.dot(yx.T, alphas)
    return w.tolist()

def simple_smo(dataset, labels, C, max_iter): #简化版SMO算法实现，未使用启发式方法对alpha对进行选择，max_iter为外层循环最大迭代次数

    dataset = np.array(dataset)
    m, n = dataset.shape
    labels = np.array(labels)
    # 初始化参数
    alphas = np.zeros(m)
    b = 0
    it = 0
    def f(x): #SVM分类器函数 y = w^Tx + b
        x = np.matrix(x).T
        data = np.matrix(dataset)
        ks = data * x
        wx = np.matrix(alphas * labels) * ks
        fx = wx + b
        return fx[0, 0]

    while it < max_iter:
        pair_changed = 0
        for i in range(m):
            a_i, x_i, y_i = alphas[i], dataset[i], labels[i]
            fx_i = f(x_i)
            E_i = fx_i - y_i
            j = select_j(i, m)
            a_j, x_j, y_j = alphas[j], dataset[j], labels[j]
            fx_j = f(x_j)
            E_j = fx_j - y_j
            K_ii, K_jj, K_ij = np.dot(x_i, x_i), np.dot(x_j, x_j), np.dot(x_i, x_j) #线性核

            # K_ii = GaussianKernelFunction(x_i, x_i) #高斯核
            # K_jj = GaussianKernelFunction(x_j, x_j)
            # K_ij = GaussianKernelFunction(x_i, x_j)

            # K_ii = PolynomialKernelFunction(x_i, x_i, 3) #多项式核
            # K_jj = PolynomialKernelFunction(x_j, x_j, 3)
            # K_ij = PolynomialKernelFunction(x_i, x_j, 3)
            
            eta = K_ii + K_jj - 2 * K_ij
            if eta <= 0:
                continue
            # 获取更新的alpha对
            a_i_old, a_j_old = a_i, a_j
            a_j_new = a_j_old + y_j * (E_i - E_j) / eta
            # 对alpha进行修剪
            if y_i != y_j:
                L = max(0, a_j_old - a_i_old)
                H = min(C, C + a_j_old - a_i_old)
            else:
                L = max(0, a_i_old + a_j_old - C)
                H = min(C, a_j_old + a_i_old)
            a_j_new = clip(a_j_new, L, H)
            a_i_new = a_i_old + y_i*y_j*(a_j_old - a_j_new)
            if abs(a_j_new - a_j_old) < 0.00001:
                continue
            alphas[i], alphas[j] = a_i_new, a_j_new
            # 更新阈值b
            b_i = -E_i - y_i * K_ii * (a_i_new - a_i_old) - y_j * K_ij * (a_j_new - a_j_old) + b
            b_j = -E_j - y_i * K_ij * (a_i_new - a_i_old) - y_j * K_jj * (a_j_new - a_j_old) + b
            if 0 < a_i_new < C:
                b = b_i
            elif 0 < a_j_new < C:
                b = b_j
            else:
                b = (b_i + b_j)/2
            pair_changed += 1
        if pair_changed == 0:
            it += 1
        else:
            it = 0
    return alphas, b

dataset, labels = creatDataSet()
alphas, b = simple_smo(dataset, labels, 0.6, 40)


# 分类数据点
classified_pts = {'+1': [], '-1': []}
for point, label in zip(dataset, labels):
    if label == 1.0:
        classified_pts['+1'].append(point)
    else:
        classified_pts['-1'].append(point)
fig = plt.figure()
ax = fig.add_subplot(111)
# 绘制数据点
for label, pts in classified_pts.items():
    pts = np.array(pts)
    ax.scatter(pts[:, 0], pts[:, 1], label=label)
# 绘制分割线
w = get_w(alphas, dataset, labels)
x1, _ = max(dataset, key=lambda x: x[0])
x2, _ = min(dataset, key=lambda x: x[0])
a1, a2 = w
y1, y2 = (-b - a1*x1)/a2, (-b - a1*x2)/a2
ax.plot([x1, x2], [y1, y2])
print(type(w))

# 绘制支持向量
for i, alpha in enumerate(alphas):
    if abs(alpha) > 1e-3:
        x, y = dataset[i]
        ax.scatter([x], [y], s=150, c='none', alpha=0.7,
                    linewidth=1.5, edgecolor='#AB3319')
plt.show()