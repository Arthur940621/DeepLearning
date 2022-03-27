import os
import gzip
import numpy as np
def load_data(data_folder, data_name, label_name):
    """
        data_folder:文件目录
        data_name:数据文件名
        label_name:标签数据文件名
    """
    with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath: # rb表示的是读取二进制数据
        y = np.frombuffer(lbpath.read(), np.uint8, offset = 8)
        print(y)
    with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
        x = np.frombuffer(imgpath.read(), np.uint8, offset = 16).reshape(len(y), 28, 28)
    return (x, y)