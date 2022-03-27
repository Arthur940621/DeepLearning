import numpy as np
import random
import matplotlib.pyplot as plt

def creatDataSet():
    dataset = []
    labels = []
    DATA = [
        [0.697,0.46,1],
        [0.774,0.376,1],
        [0.634,0.264,1],
        [0.608,0.318,1],
        [0.556,0.215,1],
        [0.403,0.237,1],
        [0.481,0.149,1],
        [0.437,0.211,1],
        [0.666,0.091,-1],
        [0.243,0.267,-1],
        [0.245,0.057,-1],
        [0.343,0.099,-1],
        [0.639,0.161,-1],
        [0.657,0.198,-1],
        [0.36,0.37,-1],
        [0.593,0.042,-1],
        [0.719,0.103,-1],
    ]
    for data in DATA:
        dataset.append(data[:-1])
        labels.append(data[-1])
    return dataset, labels



dataset, labels = creatDataSet()



# # 分类数据点
# classified_pts = {'+1': [], '-1': []}
# for point, label in zip(dataset, labels):
#     if label == 1.0:
#         classified_pts['+1'].append(point)
#     else:
#         classified_pts['-1'].append(point)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # 绘制数据点
# for label, pts in classified_pts.items():
#     pts = np.array(pts)
#     ax.scatter(pts[:, 0], pts[:, 1], label=label)

# plt.show()