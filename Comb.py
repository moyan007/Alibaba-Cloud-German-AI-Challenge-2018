import pandas as pd
import numpy as np
import h5py
import cv2

def comb_img(img_a,img_b):
    """
    将实部虚部结合
    :param img_a: 实部   Format:(32,32)
    :param img_b: 虚部   Format:(32,32)
    :return: 二者平方和开根号得到的图像    Format:(32,32)
    """
    output = np.zeros(img_a.shape)
    for i in range(img_a.shape[0]):
        for j in range(img_a.shape[1]):
            output[i][j] = np.sqrt(img_a[i][j]*img_a[i][j] + img_b[i][j]*img_b[i][j])
    return output


def comb(data, a, b):
    """
    对于一个数据集，将两个通道进行结合
    :param data: 待处理的数据集      Format:(None,32,32,8)
    :param a: 实部通道index          int
    :param b: 虚部通道index          int
    :return: 结合后的单通道数据集     Format:(None,32,32)
    """
    output = np.zeros((data.shape[0], 32, 32))
    for i in range(data.shape[0]):
        sample = data[i]
        s_a = sample[:, :, a]
        s_b = sample[:, :, b]
        output[i] = comb_img(s_a, s_b)
    return output

filename_train = 'F:/Comptition/German AI/data/training.h5'
f = h5py.File(filename_train,'r')

s1_train = f['sen1']
s2_train = f['sen2']
label_train = f['label']

s1_train=np.array(s1_train[:7,:,:,:])

print(s1_train.shape)
s1_train_7_8=comb(s1_train,6,7)

x=[]

for i in range(s1_train.shape[0]):
    pic_1 = cv2.merge([s1_train[i, :, :, 4], s1_train[i, :, :, 5], s1_train_7_8[i, :, :]])
    pic_1=np.array(pic_1)       #pic_1的shape是32*32*3
    temp=pic_1.flatten()
    x.append(temp)

x=np.array(x)


print(x.shape)
np.save('S1_data.npy',x)        #x的维数是n*(32*32*3)，使用的时候使用，np.load(),然后reshape（，32，32,3）



print("测试代码，.npy文件的使用方法")
print("====================================================")

train=np.load("S1_data.npy")
print(train.shape)
train_reshape=train.reshape((-1,32,32,3))
print(train_reshape.shape)
