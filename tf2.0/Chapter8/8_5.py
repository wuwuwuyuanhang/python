# @Auther : wuwuwu 
# @Time : 2020/2/7 
# @File : 8_5.py
# @Description : 加载ResNet50网络模型

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential

# 加载ResNet50模型，include_top为False去掉最后一层
resnet = keras.applications.ResNet50(weights='imagenet', include_top=False)

# 新建池化层
global_average_layer = layers.GlobalAveragePooling2D()

# 新建全连接层
fc = layers.Dense(100)

# 将三层连接起来
myNet = Sequential([resnet, global_average_layer, fc])
# 不训练ResNet50网络的参数
resnet.trainable = False

myNet.summary()