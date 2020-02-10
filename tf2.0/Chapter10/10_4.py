# @Auther : wuwuwu 
# @Time : 2020/2/8 
# @File : 10_4.py
# @Description : LeNet-5实战

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, datasets, losses, optimizers, metrics


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    return x, y


(x, y), (x_test, y_test) = datasets.mnist.load_data()
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess).batch(128)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(1000).map(preprocess).batch(128)

network = Sequential([
    layers.Conv2D(6, kernel_size=3, strides=1),  # 第一个卷积层，6个3*3卷积核
    layers.MaxPooling2D(pool_size=2, strides=2),  # 高宽各减半的池化层
    layers.ReLU(),  # 激活函数
    layers.Conv2D(16, kernel_size=3, strides=1),
    layers.MaxPooling2D(pool_size=2, strides=2),
    layers.ReLU(),
    layers.Flatten(),  # 打平层，方便全连接层处理
    layers.Dense(120, activation='relu'),  # 全连接层，120个节点
    layers.Dense(84, activation='relu'),
    layers.Dense(10)
])

network.build(input_shape=(None, 28, 28, 1))

# 导入误差计算，优化器模块
criten = losses.CategoricalCrossentropy(from_logits=True)
optimizer = optimizers.SGD(learning_rate=0.01)

accs = metrics.Accuracy()

for epoch in range(20):
    for step, (x, y) in enumerate(train_db):

        with tf.GradientTape() as tape:
            # 插入通道数, => [b, 28, 28, 1]
            x = tf.expand_dims(x, axis=3)
            out = network(x)
            y_onehot = tf.one_hot(y, depth=10)
            loss = criten(y_onehot, out)

        grads = tape.gradient(loss, network.trainable_variables)
        optimizer.apply_gradients(zip(grads, network.trainable_variables))
        accs.update_state(y, tf.argmax(out, axis=1))

        if step % 80 == 0:
            print(epoch, step, 'loss:', float(loss), 'accs:', accs.result().numpy())
            accs.reset_states()

        if step % 80 == 0:
            correct, total = 0, 0
            for x, y in test_db:
                x = tf.expand_dims(x, axis=3)
                out = network(x)
                pred = tf.argmax(out, axis=1)
                y = tf.cast(y, dtype=tf.int64)
                correct += float(tf.reduce_mean(tf.cast(tf.equal(pred, y), dtype=tf.float32)))
                total += x.shape[0]

            print('test acc:', correct / total)

network.save_weights('weights.ckpt')