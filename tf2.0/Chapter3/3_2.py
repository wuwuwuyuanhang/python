# @Auther : wuwuwu 
# @Time : 2020/1/23 
# @File : 3_2.py
# @Description : 模型构建

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, metrics
from matplotlib import pyplot as plt

# 设置GPU使用方式
# 获取GPU列表
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU为增长式占用
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 打印异常
        print(e)

# 自动在线下载MNIST数据集，并转为Numpy数组格式
(x, y), (x_val, y_val) = datasets.mnist.load_data() # 加载MNIST数据集
x = 2 * tf.convert_to_tensor(x, dtype=tf.float32) / 255. - 1 # 转化为浮点张量，并缩放到-1到1
y = tf.convert_to_tensor(y, dtype=tf.int32) # 转化为整型张量
print(x.shape, y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y)) # 构建数据集对象
train_dataset = train_dataset.batch(32).repeat(30) # 批量训练，每组训练30次

# 利用Sequential容器封装3个网络层，前网络层的输出默认为下层的输入
model = keras.Sequential([  # 3个非线性层的嵌套模型
    layers.Dense(256, activation='relu'), # 隐藏层1
    layers.Dense(128, activation='relu'), # 隐藏层2
    layers.Dense(10)    # 输出层，输出节点数位10
])  #直接调用这个模型对象model(x)就可以返回模型最后一层的输出o

# 随机梯度下降优化
optimizer = optimizers.SGD(lr=0.01)
acc_meter = metrics.Accuracy()

# 保存每次损失函数值
losses = []

# 训练模型
for epoch in range(20): # 训练20轮
    for step, (x, y) in enumerate(train_dataset):
        # GradientTape可以自动监视在上下文中访问的所有可训练变量
        with tf.GradientTape() as tape: # 构建梯度记录环境
            # 打平操作，[b, 28, 28] => [b, 784]
            x = tf.reshape(x, [-1, 28*28])
            # Step1. 得到模型输出output [b, 784] => [b, 10]
            out = model(x)
            # [b] => [b, 10]
            y_onehot = tf.one_hot(y, depth=10)
            # 计算差的平方和，[b, 10]
            loss = tf.square(out - y_onehot)
            # 计算每个样本的平均方差
            loss = tf.reduce_mean(loss) / x.shape[0]
            
        acc_meter.update_state(tf.argmax(out, axis=1), y)
        # Step3. 计算参数的梯度
        grads = tape.gradient(loss, model.trainable_variables)
        # w' = w - lr * grad, 更新网络参数
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 200 == 0:

            print(step, 'loss:', float(loss), 'acc:', acc_meter.result().numpy())
            acc_meter.reset_states()

    losses.append(loss)

# 绘图
plt.figure()
plt.plot(losses, color='r', marker='+', label='训练')
plt.show()