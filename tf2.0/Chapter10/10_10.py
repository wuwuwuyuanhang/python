# @Auther : wuwuwu 
# @Time : 2020/2/12 
# @File : 10_10.py
# @Description : CIFAR10与VGG13实战

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics, Sequential, losses, datasets, optimizers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def preprocess(x, y):

    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)

    return x, y

(x, y), (x_test, y_test) = datasets.cifar10.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(10000).map(preprocess).batch(128)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(64)

# 将网络实现为2个子网络：卷积子网络与全连接子网络
conv_layers = [
    # 单元1 64个3*3卷积核，输入输出同大小
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    # 高宽减半
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    # 单元2，输出通道提升至128，高宽大小减半
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    # 单元3，输出通道提升道256，高宽大小减半
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    # 单元4，输出通道提升道512，高宽大小减半
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    # 单元5，输出通道提升道512，高宽大小减半
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same")
]

conv_net = Sequential(conv_layers)

fc_net = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.relu)
])

conv_net.build(input_shape=[None, 32, 32, 3])
fc_net.build(input_shape=[None, 512])

optimizer = optimizers.Adam(lr=1e-4)

# 列表合并，合并2个子网络的参数
variables = conv_net.trainable_variables + fc_net.trainable_variables

for epoch in range(50):

    for step, (x, y) in enumerate(train_db):

        with tf.GradientTape() as tape:
            # [b, 32, 32, 3] => [b, 1, 1, 512]
            out = conv_net(x)
            # flatten, => [b, 512]
            out = tf.reshape(out, [-1, 512])
            # [b, 512] => [b, 10]
            logits = fc_net(out)
            # [b] => [b, 10]
            y_onehot = tf.one_hot(y, depth=10)
            # compute loss
            loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

        if step % 100 == 0:
            print(epoch, step, 'loss:', float(loss))

    total_num = 0
    total_correct = 0
    for x, y in test_db:
        out = conv_net(x)
        out = tf.reshape(out, [-1, 512])
        logits = fc_net(out)
        prob = tf.nn.softmax(logits, axis=1)
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)

        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)

        total_num += x.shape[0]
        total_correct += int(correct)

    acc = total_correct / total_num
    print(epoch, 'acc:', acc)