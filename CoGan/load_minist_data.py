import os
import gzip
import numpy as np

def load_data(data_folder, data_name):
    """
        data_folder:文件目录
        data_name:数据文件名
        label_name:标签数据文件名
    """
    with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
        x = np.frombuffer(imgpath.read(), np.uint8, offset = 16).reshape(60000, 28, 28)
    return x

def load_invert_mnist(data_folder, data_name):
    trX = load_data(data_folder, data_name)
    invert_trX = 255 - trX
    invert_trX = invert_trX.astype(np.uint8)
    return invert_trX
