#encoding:utf-8
import h5py
import numpy as np
import matplotlib.pyplot as plt

filename_train = 'E:/Alibaba German AI Challenge/origin_DATA/training.h5'
f = h5py.File(filename_train,'r')
s1_train = f['sen1']
s2_train = f['sen2']
label_train = f['label']

filename_vali = 'E:/Alibaba German AI Challenge/origin_DATA/validation.h5'
f = h5py.File(filename_vali,'r')
s1_vali = f['sen1']
s2_vali = f['sen2']
label_vali = f['label']

sum_train = np.sum(label_train, axis=0)
sum_vali = np.sum(label_vali, axis=0)

count = []
num = 3000
for i in range(label_train.shape[1]):
    if sum_vali[i] >= num:
        count.append(0)
    else:
        min_num = min(num,sum_train[i]+sum_vali[i])
        count.append(min_num-sum_vali[i])
#每一类需要增加的样本数量
count = np.array(count)
count = count.astype(np.int32)

#取得待取样本在trainDATA中的索引
didgit_label = np.argmax(label_train, 1)

ID = []
for index in range(17):
    c = 0
    for i in range(didgit_label.shape[0]):
        if c == count[index]:
            break
        if(didgit_label[i] == index):
            ID.append(i)
            c += 1

add_label = label_train[ID,:]
add_1 = s1_train[ID]
add_2 = s2_train[ID]
add_1 = np.array(add_1)
add_2 = np.array(add_2)
y = np.array(add_label)

#将S1和S2都考虑进来，将训练集中抽出来的数据长成n*（32*32*17）维的向量，要使用的时候直接reshape成n*32*32*17的高维数据
#如果只考虑S2的话，则不考虑S1即可
x = []
for i in range(0,add_1.shape[0]):
    temp1 = add_1[i].flatten()
    temp2 = add_2[i].flatten()
    temp = np.hstack((temp1,temp2))
    x.append(temp)
x = np.array(x)
add_data = np.hstack((x,y))
np.save("add_data.npy",add_data)


import numpy as np
import h5py
filename = 'E:/Alibaba German AI Challenge/origin_DATA/validation.h5'
f = h5py.File(filename,'r')
print('Get the h5 file')
s1 = np.array(f['sen1'])
s2 = np.array(f['sen2'])
y = np.array(f['label'])
#对验证集做同样的操作
x = []
for i in range(0,s1.shape[0]):
    temp1 = s1[i].flatten()
    temp2 = s2[i].flatten()
    temp = np.hstack((temp1,temp2))
    x.append(temp)
x = np.array(x)
data = np.hstack((x,y))
np.save('vali_data.npy',data)

#将验证集和添加的训练集结合起来，得到最后的数据集。这里注意得到的是n*(32*32*17)的数据，送入CNN的时候需要将其reshape成n*32*32*17的数据。
import numpy as np
import matplotlib.pyplot as plt
vali_data = np.load('vali_data.npy')
add_data = np.load('add_data.npy')
data = np.vstack((vali_data,add_data))

np.save('data.npy',data)