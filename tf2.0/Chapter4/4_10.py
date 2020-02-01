# @Auther : wuwuwu 
# @Time : 2020/1/28 
# @File : 4_10.py
# @Description : 前向传播

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, metrics
import matplotlib
from matplotlib import pyplot as plt

# Default parameters for plots
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus']=False

# 构建三层神经网络
# out = ReLU{ ReLU{ ReLU[ X @ W1 + b1 ] @ W2 + b2 } @ W3 + b3 }

(x, y), _ = datasets.mnist.load_data() # 加载MNIST数据集

# 将数据集创焕为tensor张量类型
x = 2 * tf.convert_to_tensor(x, dtype=tf.float32) / 255. - 1
y = tf.convert_to_tensor(y, dtype=tf.int32)

# 构建训练集对象
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.batch(128).repeat(10) # 批量训练

lr = 0.01
acc_meter = metrics.Accuracy()
losses = []

# 每层的张量都是需要被优化的，故使用Variable类型，并使用截断的正态分布初始化权值张量
# 偏置向量初始值为0即可
# 第一层参数
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))

# 第二层参数
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))

# 第三层参数
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

for epoch in range(20):
    for step, (x, y) in enumerate(train_db):
        # [b, 28, 28] => [b, 28*28]
        x = tf.reshape(x, [-1, 28 * 28])

        with tf.GradientTape() as tape:
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)

            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)

            out = h2 @ w3 + b3

            y_onehot = tf.one_hot(y, depth=10)
            # 计算mse
            loss = tf.square(y_onehot - out)
            loss = tf.reduce_mean(loss)

        # 自动梯度，需要求梯度的张量有[w1, b1, w2, b2, w3, b3]
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # 梯度更新，assign_sub将当前值减去参数值，原地更新
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        acc_meter.update_state(tf.argmax(out, axis=1), y)

        if step % 100 == 0:
            print(step, 'loss:', float(loss), 'acc:', acc_meter.result().numpy())
            acc_meter.reset_states()

    losses.append(float(loss))

plt.figure()
plt.plot(losses, color='r', marker='+', label='训练')
plt.show()