import numpy as np
import os

name = 't10k-images-idx3-ubyte'
data_dir = 'data/mnist/'
save_dir = os.path.join(data_dir, 'invert')
os.makedirs(save_dir, exist_ok=True)
fd = open(data_dir+name)
data = np.fromfile(file=fd,dtype=np.uint8)
trX = data[16:].reshape((-1,28,28)).astype(np.int)
invert_trX = 255-trX

np.save(os.path.join(save_dir, name), invert_trX)

name = 'train-images-idx3-ubyte'
fd = open(data_dir+name)
data = np.fromfile(file=fd,dtype=np.uint8)
trX = data[16:].reshape((-1,28,28)).astype(np.int)
invert_trX = 255-trX
np.save(os.path.join(save_dir, name), invert_trX)