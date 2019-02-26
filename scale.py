#encoding:utf-8
'''
对数据按通道进行归一化¶
scaled_x = (x-min)/(max-min)
其中max、min分别是该像素所在通道的最大值最小值
'''
import numpy as np
data = np.load('C:/Users/Jack-Gao/Desktop/Alibaba-German-AI-Challenge-master/project/S1_data.npy')

print(data.max())#5.2481136322
print(data.min())#1.6921524803e-05
#data的维度是n*(32*32*3)
x = data[:,:]


# 一个通道对应的列数
pix_num_channel = 32*32

for i in range(3):
    temp = x[:,i*pix_num_channel:(i+1)*pix_num_channel]
    temp_max = temp.max()
    temp_min = temp.min()
    if temp_max == temp_min:
        base = 1
    else:
         base = temp_max-temp_min
    for row in range(data.shape[0]):
        for col in range(i*pix_num_channel,(i+1)*pix_num_channel):
            data[row][col] = (data[row][col]-temp_min)/base

np.save('data_scaled.npy',data)
print("finished")


'''测试代码，归一化成功'''
print(data[0])
print(data[0].max())    #0.383713250981
print(data[0].min())    #8.04174320135e-05