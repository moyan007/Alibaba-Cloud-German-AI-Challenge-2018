#encoding:utf-8
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename = 'E:/Alibaba German AI Challenge/origin_DATA/validation.h5'
f = h5py.File(filename,'r')
s1 = f['sen1']
s2 = f['sen2']
label = f['label']

label_qty = np.sum(label, axis=0)
x = list(range(17))
plt.bar(x, label_qty, width= 0.5, color = "cornflowerblue")
my_x_ticks = np.arange(0, 17, 1)
my_y_ticks = np.arange(0, 4000, 500)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.xlim((-1, 17))
plt.ylim((0, 3500))
plt.show()