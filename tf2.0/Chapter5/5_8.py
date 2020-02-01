# @Auther : wuwuwu 
# @Time : 2020/2/1 
# @File : 5_8.py
# @Description : MNIST测试实战

import tensorflow as tf
from tensorflow import keras
import matplotlib
from matplotlib import pyplot as plt

# Default parameters for plots
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus']=False

def preprocess(x, y): # 自定义的预处理函数
    # 调用此函数时自动传入x, y对象，shape为[b, 28, 28], [b]
    # 标准化到0-1
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28*28]) # 打平
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)

    # 返回的x, y将替换传入的x, y参数，从而实现数据的预处理
    return x, y

# 加载数据集
(x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 批训练长度
batch_size = 512
train_db = tf.data.Dataset.from_tensor_slices((x, y)) # 转为Dataset对象
train_db = train_db.shuffle(1000) # 随机打散
train_db = train_db.batch(batch_size) # 批训练
train_db = train_db.map(preprocess) # 预处理
train_db = train_db.repeat(20) # 内部循环20次

# 测试数据集处理
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(1000).batch(batch_size).map(preprocess)

def main():

    # 学习率
    lr = 1e-2
    accs, losses = [], []

    # 神经层参数
    w1, b1 = tf.Variable(tf.random.normal([784, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
    w2, b2 = tf.Variable(tf.random.normal([256, 128], stddev=0.1)), tf.Variable(tf.zeros([128]))
    w3, b3 = tf.Variable(tf.random.normal([128, 10], stddev=0.1)), tf.Variable(tf.zeros([10]))

    for step, (x, y) in enumerate(train_db):

        x = tf.reshape(x, (-1, 784))

        with tf.GradientTape() as tape:
            # layer1
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            # layer2
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # output
            out = h2 @ w3 + b3

            # 计算损失函数
            # [b, 10] - [b, 10]
            loss = tf.square(y - out)
            # [b, 10] => scalar
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        for p, g in zip([w1, b1, w2, b2, w3, b3], grads):
            p.assign_sub(lr * g)

        if step % 80 == 0:
            print(step, 'loss: ', float(loss))
            losses.append(float(loss))

        if step %80 == 0:
            # evaluate/test
            total, total_correct = 0., 0

            for x, y in test_db:
                h1 = tf.nn.relu(x @ w1 + b1)
                h2 = tf.nn.relu(h1 @ w2 + b2)
                out = h2 @ w3 + b3
                # [b, 10] => [b]
                pred = tf.argmax(out, axis=1)
                # convert one_hot y to number y
                y = tf.argmax(y, axis=1)
                # bool type
                correct = tf.equal(pred, y)
                # bool tensor => int tensor => numpy
                total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                total += x.shape[0]

            print(step, 'Evaluate Acc:', total_correct/total)

            accs.append(total_correct/total)

    plt.figure()
    x = [i * 80 for i in range(len(losses))]
    plt.plot(x, losses, color='r', marker='+', label='训练')
    plt.ylabel('MSE')
    plt.xlabel('Step')
    #plt.show()

    plt.figure()
    plt.plot(x, accs, color='b', marker='*', label='测试')
    plt.ylabel('准确率')
    plt.xlabel('Step')
    plt.show()

if __name__ == '__main__':
    main()

