# @Auther : wuwuwu 
# @Time : 2020/2/2 
# @File : 6_8.py
# @Description : 汽车油耗预测实战

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from matplotlib import pyplot as plt

dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

# 效能（公里数每加仑），气缸数，排量，马力，重量
# 加速度，型号年份，产地
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()

# 统计空白数据,并清除
dataset.isna().sum() # 统计空白数据
dataset = dataset.dropna() # 删除空白数据项
dataset.isna().sum() # 再次统计空白项

# 处理类别型数据，其中origin列代表了类别1,2,3,分布代表产地：美国、欧洲、日本
# 其弹出（删除并返回）这一列
origin = dataset.pop('Origin')
# 根据origin列来写入新列
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0

# 切分为训练集和测试集
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# 查看训练集的输入X的统计数据
train_stats = train_dataset.describe()
train_stats.pop("MPG") # 仅保留输入x
train_stats = train_stats.transpose() # 转置

# 移动MPG油耗效能这一列为真实标签Y
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# 标准化数据
def norm(x): # 减去每个字段的均值，并除以标准差
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset) # 标准化训练集
normed_test_data = norm(test_dataset) # 标准化测试集

train_db = tf.data.Dataset.from_tensor_slices((normed_train_data.values, train_labels.values))
train_db = train_db.shuffle(1000).batch(32) # 随机打散，批量化

class NetWork(keras.Model):
    # 回归网络
    def __init__(self):
        super(NetWork, self).__init__()
        # 3个全连接神经层
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(1, activation=None)

    def call(self, inputs, training=None, mask=None):
        # 依次通过3个全连接层
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

model = NetWork() # 创建网络类实列
# 通过build函数完成内部张量的创建，其中4为任意设置的batch数量，9为输入特征长度
model.build(input_shape=(None, 9))
model.summary() # 打印网络信息

optimizer = tf.keras.optimizers.RMSprop(0.001) # 创建优化器，指定学习率

train_mae_losses = []
test_mae_losses = []

for epoch in range(200): # 200个Epoch
    for step, (x, y) in enumerate(train_db): # 遍历一次训练集
        # 梯度记录器
        with tf.GradientTape() as tape:
            out = model(x) # 通过网络获得输出
            loss = tf.reduce_mean(tf.keras.losses.MSE(y, out)) # 计算MSE
            mae_loss = tf.reduce_mean(tf.keras.losses.MAE(y, out)) # 计算MAE

        if step % 10 == 0:
            print(epoch, step, float(loss))

        # 计算梯度并更新
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_mae_losses.append(float(mae_loss))
    out = model(tf.constant(normed_test_data.values))
    test_mae_losses.append(tf.reduce_mean(tf.keras.losses.MAE(test_labels, out)))

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.plot(train_mae_losses, label='Train')

plt.plot(test_mae_losses, label='Test')
plt.legend()

plt.legend()
plt.show()