# @Auther : wuwuwu 
# @Time : 2020/2/4 
# @File : 7_9_2.py
# @Description : 使用tensorflow实现反向传播实现

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, metrics
from matplotlib import pyplot as plt

# 利用scikit-learn提供的make_moons函数生成2000个线性不可分割的2分类数据集
N_SAMPLES = 2000 # 采样点数
TEST_SIZE = 0.3 # 测试数据比例
X, y = make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=100)
# 将2000个点按照7：3分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

model = Sequential([
    layers.Dense(25, activation='sigmoid'), # 隐藏层1, 2 => 25
    layers.Dense(50, activation='sigmoid'), # 隐藏层2, 25 => 50
    layers.Dense(25, activation='sigmoid'), # 隐藏层3, 50 => 25
    layers.Dense(2, activation=None) # 输出层, 25 => 2
])

model.build(input_shape=(None, 2))
model.summary()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

train_db = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_db = train_db.batch(32) # 单个训练

train_mses, test_mses = [], []
train_accs = metrics.SparseCategoricalAccuracy()
accs = []

for epoch in range(200):
    for step, (x, y) in enumerate(train_db):

        y_onehot = tf.one_hot(y, depth=2)

        with tf.GradientTape() as tape:
            out = model(x)

            loss = tf.keras.losses.categorical_crossentropy(y_onehot, out, from_logits=True)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_accs.update_state(y, out)

    if epoch % 10 == 0:
        print(epoch, step, 'loss:', float(loss), 'acc:', train_accs.result().numpy())
        train_mses.append(float(loss))
        accs.append(train_accs.result().numpy())
        train_accs.reset_states()


    if epoch % 10 == 0:
        test_out = model(tf.constant(X_test))
        y_test_onehot = tf.constant(y_test)
        y_test_onehot = tf.one_hot(y_test_onehot, depth=2)
        test_loss = tf.keras.losses.categorical_crossentropy(y_test_onehot, test_out, from_logits=True)
        test_mses.append(float(tf.reduce_mean(test_loss)))


plt.figure()
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('sigmoid_mses')
plt.plot(train_mses, label='train mse')

plt.plot(test_mses, label='test mse')

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('ACC')
plt.title('sigmoid accs')
plt.plot(accs, label='train acc')

plt.show()