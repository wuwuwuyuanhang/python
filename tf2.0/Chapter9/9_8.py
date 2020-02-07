# @Auther : wuwuwu 
# @Time : 2020/2/7 
# @File : 9_8.py
# @Description : 过拟合问题实战

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics, Sequential, regularizers
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False

N_SAMPLE = 1000
TEST_SIZE = 0.2
X, y = make_moons(n_samples=N_SAMPLE, noise=0.25, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

def make_plot(X, y, plot_name, XX=None, YY=None, preds=None):
    plt.figure()
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(plot_name)
    if(XX is not None and YY is not None and preds is not None):
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha=0.8, cmap=plt.cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5], camp="Greys", vmin=0, vmax=6)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral, edgecolors='none')
    plt.show()

make_plot(X, y, "数据")

xx = np.arange(-2, 3, 0.1)
yy = np.arange(-1.5, 2, 0.1)
XX, YY = np.meshgrid(xx, yy)

# 网络层数的影响
for n in range(5): # 创建5种不同层数的网络
    model = Sequential() # 创建容器
    # 创建第一层
    model.add(layers.Dense(8, input_dim=2, activation='relu'))
    for _ in range(n): # 添加n层，共n+2层
        model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid')) # 创建最末层
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # 模型装配与训练
    history = model.fit(X_train, y_train, epochs=20, verbose=1)
    # 绘制不同层数的网络决策边界曲线
    preds = model.predict_classes(np.c_[XX.ravel(), YY.ravel()])
    title = "网络层数({})".format(n)
    make_plot(X_train, y_train, title, XX, YY, preds)

# Dropout的影响
for n in range(5): # 构建5种不同数量Dropout层的网络
    model = Sequential()
    model.add(layers.Dense(8, input_dim=2, activation='relu'))
    counter = 0
    for _ in range(n):
        model.add(layers.Dense(64, activation='relu'))
        if counter < n: # 添加n个Dropout层
            counter += 1
            model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=20, verbose=1)
    preds = model.predict_classes(np.c_[XX.ravel(), YY.ravel()])
    title = "Dropout({})".format(n)
    make_plot(X_train, y_train, title, XX, YY, preds)

def build_model_with_regularizeation(_lambda):
    # 创建带正则化项的神经网络
    model = Sequential()
    model.add(layers.Dense(8, input_dim=2, activation='relu'))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(_lambda)))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(_lambda)))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(_lambda)))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

for _lambda in [1e-5, 1e-3, 1e-1, 0.12, 0.13]: # 设置不同的正则化系数
    model = build_model_with_regularizeation(_lambda)
    history = model.fit(X_train, y_train, epochs=20, verbose=1)
    plot_title = "正则化-[lambda = {}]".format(str(_lambda))
    preds = model.predict_classes(np.c_[XX.ravel(), YY.ravel()])
    title = "正则化".format(_lambda)
    make_plot(X_train, y_train, title, XX, YY, preds)