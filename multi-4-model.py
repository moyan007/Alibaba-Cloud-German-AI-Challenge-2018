import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES']='2,3'

base_dir = "../data/"
path_training = os.path.join(base_dir, "training.h5")
path_validation = os.path.join(base_dir, "validation.h5")
path_test = os.path.join(base_dir, "round2_test_b_20190211.h5")

fid_training = h5py.File(path_training,'r')
fid_validation = h5py.File(path_validation,'r')
fid_test = h5py.File(path_test,'r')

print("shape for each channel.")#只有三个字典参数
s1_training = fid_training['sen1']
print(s1_training.shape)#(352366, 32, 32, 8)
s2_training = fid_training['sen2']
print(s2_training.shape)#(352366, 32, 32, 10)
label_training = fid_training['label']
print(label_training.shape)#(352366, 17)

print("-" * 60)
print("validation part")
s1_validation = fid_validation['sen1']
print(s1_validation.shape)#(24119, 32, 32, 8)
s2_validation = fid_validation['sen2']
print(s2_validation.shape)#(24119, 32, 32, 10)
label_validation = fid_validation['label']
print(label_validation.shape)#(24119, 17)

print("-" * 60)
print("test part")
s1_test = fid_test['sen1']
print(s1_test.shape)#(4838, 32, 32, 8)
s2_test = fid_test['sen2']
print(s2_test.shape)#(4838, 32, 32, 10)

"""data processing"""
print("-" * 60)
print(" data pre-processing part")

# train_s2 = np.array(fid_training['sen2'])
# train_label = np.array(fid_training['label'])
#
# validation_s2 = np.array(fid_validation['sen2'])
# validation_label = np.array(fid_validation['label'])
#
# train_data = np.vstack([train_s2, validation_s2])
# train_label = np.vstack([train_label, validation_label])

x_test = np.array(fid_test['sen2'])

# from sklearn.model_selection import train_test_split
# x_train, x_val, y_train, y_val = train_test_split(train_data, train_label,
#         test_size=0.2, shuffle=True, stratify=train_label)

print("-" * 60)
print(" build and train the model")
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Conv2D, Activation
from keras.layers import add, Flatten, Dropout
from keras.optimizers import RMSprop,Adam
from keras.utils import multi_gpu_model
from keras.initializers import RandomNormal
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras import models
seed = 7
np.random.seed(seed)

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding,kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
               kernel_regularizer=regularizers.l2(0.01),strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(name=bn_name)(x)
    return x


def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter[0], kernel_size=(1, 1), strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(3, 3), padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1, 1), padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter[2], strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


inpt = Input(shape=(32, 32, 10))

x = Conv2d_BN(inpt, nb_filter=96, kernel_size=(3, 3), strides=(1, 1), padding='same')

x = Conv_Block(x, nb_filter=[64, 64, 128], kernel_size=(3, 3), strides=(1, 1), with_conv_shortcut=True)
x = Conv_Block(x, nb_filter=[64, 64, 128], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[64, 64, 128], kernel_size=(3, 3))

x = Conv_Block(x, nb_filter=[96, 96, 160], kernel_size=(3, 3), strides=(1, 1), with_conv_shortcut=True)
x = Conv_Block(x, nb_filter=[96, 96, 160], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[96, 96, 160], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[96, 96, 160], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[96, 96, 160], kernel_size=(3, 3))
# x = Conv_Block(x, nb_filter=[96, 96, 160], kernel_size=(3, 3))

x = Conv_Block(x, nb_filter=[112, 112, 192], kernel_size=(3, 3), strides=(1, 1), with_conv_shortcut=True)
x = Conv_Block(x, nb_filter=[112, 112, 192], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[112, 112, 192], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[112, 112, 192], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[112, 112, 192], kernel_size=(3, 3))
# x = Conv_Block(x, nb_filter=[112, 112, 192], kernel_size=(3, 3))

x = Conv_Block(x, nb_filter=[128, 128, 224], kernel_size=(3, 3), strides=(1, 1), with_conv_shortcut=True)
x = Conv_Block(x, nb_filter=[128, 128, 224], kernel_size=(3, 3))
x = Conv_Block(x, nb_filter=[128, 128, 224], kernel_size=(3, 3))

x = Flatten()(x)
x = Dense(512, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(0.01),
          activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(0.01),
          activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(17, activation='softmax')(x)

model = Model(inputs=inpt, outputs=x)
model = multi_gpu_model(model, gpus=2)  #add

# checkpoint = ModelCheckpoint('epochResnet50_adam128_l20001.h5',
#                              monitor='val_acc', save_weights_only= True, save_best_only=True, verbose=1,period=5)
# reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5,  patience=3, verbose=1)#,min_lr= 1e-8,
# early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1)
#
# model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
model.summary()
model.load_weights('epochResnet50_adam128_l20001.h5')
# model.fit(x_train,y_train,epochs=150,batch_size=128,shuffle=True,validation_data=(x_val, y_val),
#           callbacks=[checkpoint,reduce_lr,early_stopping],verbose=1)

"""output"""
pred1 = model.predict(x_test)

seed = 7
np.random.seed(seed)
"""model resnet34"""
def Conv2d_BN_1(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(0.001),
               name=conv_name)(x)
    x = BatchNormalization(name=bn_name)(x)
    return x

def Conv_Block_1(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN_1(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN_1(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN_1(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

weight_decay = 0.001#之前版本的0.0005
inpt = Input(shape=(32, 32, 10))
x = Conv2d_BN_1(inpt, nb_filter=64, kernel_size=(3, 3), padding='same')
# (32,32,64)
x = Conv_Block_1(x, nb_filter=64, kernel_size=(3, 3), with_conv_shortcut=True)
x = Conv_Block_1(x, nb_filter=64, kernel_size=(3, 3))
x = Conv_Block_1(x, nb_filter=64, kernel_size=(3, 3))
x = Conv_Block_1(x, nb_filter=64, kernel_size=(3, 3))
# (28,28,128)
x = Conv_Block_1(x, nb_filter=96, kernel_size=(3, 3), with_conv_shortcut=True)
x = Conv_Block_1(x, nb_filter=96, kernel_size=(3, 3))
x = Conv_Block_1(x, nb_filter=96, kernel_size=(3, 3))
x = Conv_Block_1(x, nb_filter=96, kernel_size=(3, 3))
x = Conv_Block_1(x, nb_filter=96, kernel_size=(3, 3))
x = Conv_Block_1(x, nb_filter=96, kernel_size=(3, 3))
# (14,14,256)
x = Conv_Block_1(x, nb_filter=128, kernel_size=(3, 3), with_conv_shortcut=True)
x = Conv_Block_1(x, nb_filter=128, kernel_size=(3, 3))
x = Conv_Block_1(x, nb_filter=128, kernel_size=(3, 3))
x = Conv_Block_1(x, nb_filter=128, kernel_size=(3, 3))
x = Conv_Block_1(x, nb_filter=128, kernel_size=(3, 3))
x = Conv_Block_1(x, nb_filter=128, kernel_size=(3, 3))

# (7,7,512)
x = Conv_Block_1(x, nb_filter=144, kernel_size=(3, 3), with_conv_shortcut=True)
x = Conv_Block_1(x, nb_filter=144, kernel_size=(3, 3))
x = Conv_Block_1(x, nb_filter=144, kernel_size=(3, 3)) #176?
x = Conv_Block_1(x, nb_filter=144, kernel_size=(3, 3))

x = Flatten()(x)
x = Dense(512, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(weight_decay),
          activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(weight_decay),
          activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(17, activation='softmax')(x)

model_1 = Model(inputs=inpt, outputs=x)
model_1 = multi_gpu_model(model_1, gpus=2)  #add
model_1.load_weights('epochResnet34_adam_batch256.h5')
# model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
model_1.summary()
# model.fit(x_train,y_train,epochs=150,batch_size=128,shuffle=True,validation_data=(x_val, y_val),
#           callbacks=[checkpoint,reduce_lr,early_stopping],verbose=1)
"""output"""
pred2 = model_1.predict(x_test)

from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers import Input, Dropout, Dense, Flatten, Activation, add, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.layers.merge import Concatenate
from keras import regularizers
from keras.initializers import RandomNormal
from keras.regularizers import l2
from keras import initializers
from keras.models import Model
from keras import backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import models
import keras.backend as K


def conv2d_bn(x, nb_filter, num_row, num_col,
              padding='same', strides=(1, 1), use_bias=False):

    # channel_axis = 3
    x = Convolution2D(nb_filter, (num_row, num_col),
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      kernel_regularizer=regularizers.l2(0.001),
                      kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    x = BatchNormalization()(x) #axis=channel_axis
    x = Activation('relu')(x)
    return x

def block_inception_a(input):

    channel_axis = 3
    branch_0 = conv2d_bn(input, 160, 1, 1)

    branch_1 = conv2d_bn(input, 96, 1, 1)
    branch_1 = conv2d_bn(branch_1, 160, 3, 3)

    branch_2 = conv2d_bn(input, 96, 1, 1)
    branch_2 = conv2d_bn(branch_2, 128, 3, 3)
    branch_2 = conv2d_bn(branch_2, 160, 3, 3)

    # branch_3 = AveragePooling2D((2, 2), strides=(1, 1), padding='same')(input)
    # branch_3 = conv2d_bn(branch_3, 96, 1, 1)

    x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
    return x

def block_reduction_a(input):

    channel_axis = 3
    branch_0 = conv2d_bn(input, 192, 3, 3, strides=(1, 1), padding='same')

    branch_1 = conv2d_bn(input, 128, 1, 1)
    branch_1 = conv2d_bn(branch_1, 160, 3, 3)
    branch_1 = conv2d_bn(branch_1, 192, 3, 3, strides=(1, 1), padding='same')

    # branch_2 = MaxPooling2D((2, 2),strides=(2, 2), padding='same')(input)  #strides=(2, 2),

    x = concatenate([branch_0, branch_1], axis=channel_axis)
    return x

def block_inception_b(input):

    channel_axis = 3
    branch_0 = conv2d_bn(input, 224, 1, 1)

    branch_1 = conv2d_bn(input, 160, 1, 1)
    branch_1 = conv2d_bn(branch_1, 192, 1, 3)
    branch_1 = conv2d_bn(branch_1, 224, 3, 1)

    branch_2 = conv2d_bn(input, 160, 1, 1)
    branch_2 = conv2d_bn(branch_2, 160, 3, 1)
    branch_2 = conv2d_bn(branch_2, 192, 1, 3)
    branch_2 = conv2d_bn(branch_2, 192, 3, 1)
    branch_2 = conv2d_bn(branch_2, 224, 1, 3)

    # branch_3 = AveragePooling2D((2, 2), strides=(1, 1), padding='same')(input)
    # branch_3 = conv2d_bn(branch_3, 128, 1, 1)

    x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
    return x

def block_reduction_b(input):

    channel_axis = 3
    branch_0 = conv2d_bn(input, 128, 1, 1)
    branch_0 = conv2d_bn(branch_0, 128, 3, 3, strides=(1, 1), padding='same')

    branch_1 = conv2d_bn(input, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 192, 1, 3)
    branch_1 = conv2d_bn(branch_1, 256, 3, 1)
    branch_1 = conv2d_bn(branch_1, 256, 3, 3, strides=(1, 1), padding='same')

    # branch_2 = MaxPooling2D((2, 2), padding='same')(input) #, strides=(2, 2)

    x = concatenate([branch_0, branch_1], axis=channel_axis)
    return x

def block_inception_c(input):

    channel_axis = 3
    branch_0 = conv2d_bn(input, 160, 1, 1)

    branch_1 = conv2d_bn(input, 192, 1, 1)
    branch_10 = conv2d_bn(branch_1, 160, 1, 3)
    branch_11 = conv2d_bn(branch_1, 160, 3, 1)
    branch_1 = concatenate([branch_10, branch_11], axis=channel_axis)

    branch_2 = conv2d_bn(input, 192, 1, 1)
    branch_2 = conv2d_bn(branch_2, 256, 3, 1)
    branch_2 = conv2d_bn(branch_2, 384, 1, 3)
    branch_20 = conv2d_bn(branch_2, 160, 1, 3)
    branch_21 = conv2d_bn(branch_2, 160, 3, 1)
    branch_2 = concatenate([branch_20, branch_21], axis=channel_axis)

    # branch_3 = AveragePooling2D((2, 2), strides=(1, 1), padding='same')(input)
    # branch_3 = conv2d_bn(branch_3, 256, 1, 1)

    x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
    return x

def inception_v4_base(input):

    # channel_axis = 3
    net = conv2d_bn(input, 96, 3, 3, padding='same')
    net = conv2d_bn(net, 96, 3, 3, padding='same')
    net = conv2d_bn(net, 128, 3, 3, strides=(2, 2), padding='same') #96

    # 35 x 35 x 384
    # 4 x Inception-A blocks
    for idx in range(4):
        net = block_inception_a(net)

    # 35 x 35 x 384
    # Reduction-A block
    net = block_reduction_a(net)

    # 17 x 17 x 1024
    # 7 x Inception-B blocks
    for idx in range(7):
        net = block_inception_b(net)

    # 17 x 17 x 1024
    # Reduction-B block
    net = block_reduction_b(net)

    # 8 x 8 x 1536
    # 3 x Inception-C blocks
    for idx in range(3):
        net = block_inception_c(net)

    return net

def inception_v4():

    inputs = Input((32, 32, 10))
    # Make inception base
    x = inception_v4_base(inputs)

    x = Flatten()(x)
    x = Dense(512,activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                    kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(128,activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                    kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(17,activation='softmax')(x)

    model = Model(inputs, x, name='inception_v4')
    return model

def create_model():
    return inception_v4()

model2 = create_model()
model2 = multi_gpu_model(model2, gpus=2)  #add
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model2.summary()
model2.load_weights('epochInception_adam_batch128.h5')
# model.fit(x_train,y_train,epochs=150,batch_size=128,shuffle=True,validation_data=(x_val, y_val),
#           callbacks=[checkpoint,reduce_lr,early_stopping],verbose=1)   #模型忘了加权重了

"""output"""
pred3 = model2.predict(x_test)

def build_model():

    model = models.Sequential()
    weight_decay = 0.01

    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=[32,32,10],kernel_initializer= RandomNormal(mean=0.0, stddev=0.01),kernel_regularizer=regularizers.l2(weight_decay)))  #no same
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(96, (3, 3), padding='same',kernel_initializer= RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(96, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), padding='same',kernel_initializer= RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(192, (3, 3), padding='same',kernel_initializer= RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(192, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), padding='same',kernel_initializer= RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), padding='same',kernel_initializer= RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(320, (3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(512,kernel_initializer= RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128,kernel_initializer= RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(17))
    model.add(Activation('softmax'))
    return model

model3 = build_model()
model3.summary()
model3 = multi_gpu_model(model3, gpus=2)  #add
model3.load_weights('epochCNN7_adam_l2_001.h5')

pred4 = model3.predict(x_test)

pred = 0.2*pred1+0.3*pred2+0.3*pred3+0.2*pred4

pred_index = np.argmax(pred, axis=1)

from keras.utils import to_categorical
pred = to_categorical(pred_index).astype(int)
import pandas as pd

data = pd.DataFrame(pred)
data.to_csv('output_4model_resnet50Adam128Regular_resnet34Adam128Regular_Inception_adam128lr001_dataRecontribute_CNN7_adam_l2_001.csv',header=False,index=False)
