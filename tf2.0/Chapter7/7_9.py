# @Auther : wuwuwu 
# @Time : 2020/2/3 
# @File : 7_9.py
# @Description : 反向传播算法实战
# 本章节使用没有自动求导功能的Numpy，激活函数选择Sigmoid，其他激活函数不适用

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt

# 利用scikit-learn提供的make_moons函数生成2000个线性不可分割的2分类数据集
N_SAMPLES = 2000 # 采样点数
TEST_SIZE = 0.3 # 测试数据比例
X, y = make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=100)
# 将2000个点按照7：3分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

# 绘制数据集的分布，X为2D坐标，y为数据点的标签
def make_plot(X, y, plot_name):
    plt.figure(figsize=(16, 12))
    axes = plt.gca()
    axes.set(xlabel="$x_1$", ylabel="$x_2$")
    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40)
    plt.show()

class Layer:
    # 全连接网络层
    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None):
        """
        :param n_input: 输入节点数
        :param n_neurons: 输出节点数
        :param activation: 激活函数类型
        :param weights: 权值张量，默认类内部生成
        :param bias: 偏置，默认类内部生成
        """
        # 通过正态分布初始化网络权值，初始化非常重要，不恰当的初始化将导致网络不收敛
        self.weights = weights if weights is not None else \
            np.random.randn(n_input, n_neurons) * np.sqrt(1 / n_neurons)
        self.bias = bias if bias is not None else np.random.rand(n_neurons) * 0.1
        self.activation = activation # 激活函数类型，如'sigmoid'
        self.last_activation = None # 激活函数的输出值o
        self.error = None # 用于计算当前层的delta变量的中间变量
        self.delta = None # 记录当前层的delta变量，用于计算梯度

    def activate(self, x):
        # 前向传播函数
        r = np.dot(x, self.weights) + self.bias # X@w+b
        # 通过激活函数，得到全连接层的输出o
        self.last_activation = self._apply_activation(r)
        return self.last_activation

    def _apply_activation(self, r):
        # 计算激活函数的输出
        if self.activation is None:
            return r
        elif self.activation == 'relu':
            return np.maximum(r, 0)
        elif self.activation == 'tanh':
            return np.tanh(r)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))

        return r

    def apply_activation_derivative(self, r):
        # 计算激活函数的导数
        # 无激活函数，导数为1
        if self.activation is None:
            return np.ones_like(r)
        elif self.activation == 'relu':
            grad = np.array(r, copy=True)
            grad[r > 0] = 1.
            grad[r <= 0] = 0.
            return grad
        elif self.activation == 'tanh':
            return 1 - r ** 2
        elif self.activation == 'sigmoid':
            return r * (1 - r)

        return r


class NeuralNetwork:
    # 神经网络模型大类
    def __init__(self):
        self._layers = [] # 网络层对象列表

    def add_layer(self, layer):
        # 追加网络层
        self._layers.append(layer)

    def feed_forward(self, X):
        # 前向传播
        for layer in self._layers:
            # 依次通过各个网络层
            X = layer.activate(X)
        return X

    def backpropagation(self, X, y, learning_rate):
        # 反向传播算法
        # 前向计算，得到输出值
        output = self.feed_forward(X)
        for i in reversed(range(len(self._layers))): # 反向循环
            layer = self._layers[i] # 得到当前层对象
            # 如果是输层
            if layer == self._layers[-1]: # 对于输出层
                layer.error = y - output # 计算2分类任务的均方差的导数
                # 关键步骤：计算最后一层的delta，参数输出层的梯度公式
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else: # 如果是隐藏层
                next_layer = self._layers[i+1] # 得到下一层对象
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                # 关键步骤：计算隐藏层的delta，参数输出层的梯度公式
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)

        # 循环更新权值
        for i in range(len(self._layers)):
            layer = self._layers[i]
            # o_i为上一网络层的输出
            o_i = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
            # 梯度下降算法，delta是公式中的负数，故这里用加号
            layer.weights += layer.delta * o_i.T * learning_rate

    def predict(self, x):
        output = self.feed_forward(x)
        return np.argmax(output, axis=-1)

    def accuracy(self, pre, y):
        return (np.sum(pre == y) * 1.0) / len(pre)

    def train(self, X_train, X_test, y_train, y_test, learning_rate, max_epochs):
        # 网络训练函数
        y_onehot = np.zeros((y_train.shape[0], 2))
        y_onehot[np.arange(y_train.shape[0]), y_train] = 1

        mses = []
        accs = []
        for i in range(int(max_epochs)):
            for j in range(len(X_train)):
                self.backpropagation(X_train[j], y_onehot[j], learning_rate)
            if i % 10 == 0:
                mse = np.mean(np.square(y_onehot - self.feed_forward(X_train)))
                mses.append(mse)
                print('Epoch: #%s, MSE: %f' % (i, float(mse)))

                # 统计并打印准确率
                acc = (self.accuracy(self.predict(X_test), y_test.flatten()) * 100)
                accs.append(acc)
                print('Accuracy: %.2f%%' % acc)

        return mses, accs



nn = NeuralNetwork()
nn.add_layer(Layer(2, 25, 'sigmoid')) # 隐藏层1, 2=>25
nn.add_layer(Layer(25, 50, 'sigmoid')) # 隐藏层2, 25=>50
nn.add_layer(Layer(50, 25, 'sigmoid')) # 隐藏层3, 50=>25
nn.add_layer(Layer(25, 2, 'sigmoid')) # 隐藏层4, 25=>2

def main():
    #make_plot(X, y, "Classification Dataset Visualization")
    mses, accs = nn.train(X_train, X_test, y_train, y_test, 0.01, 500)
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('sigmoid_mses')
    plt.plot(mses)

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('ACCS')
    plt.title('sigmoid_accs')
    plt.plot(accs)

    plt.show()

if __name__ == '__main__':
    main()