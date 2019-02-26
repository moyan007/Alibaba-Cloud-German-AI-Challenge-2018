#encoding:utf-8
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def GS(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    step_0 = cv2.GaussianBlur(img, (5, 5), 0)
    output = cv2.filter2D(step_0, -1, kernel=kernel)
    return output


filename = 'E:/Alibaba German AI Challenge/origin_DATA/round1_test_a_20181109.h5'
f = h5py.File(filename, 'r')
s2 = np.array(f['sen2'])

print('shape of s2 is', s2.shape[0])

process_imgs = np.zeros(s2.shape)

for i in range(s2.shape[0]):
    temp = s2[i]
    for j in range(10):
        img = temp[:, :, j].reshape((32, 32))
        pro_img = GS(img)
        process_imgs[i, :, :, j] = pro_img
x = []
for i in range(0,s2.shape[0]):
    temp = s2[i].flatten()
    x.append(temp)
x = np.array(x)
np.save('test_s2_gs.npy',x)