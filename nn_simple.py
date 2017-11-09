# -*- coding: utf-8 -*-
import enum
from enum import Enum

import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


@enum.unique
class ACTIVE_TYPE(Enum):
    sigmoid = 1
    tanh = 2


ACTIVE_FUNC_MAP = {
    ACTIVE_TYPE.sigmoid: (sigmoid, sigmoid_deriv),
    ACTIVE_TYPE.tanh: (tanh, tanh_deriv),
}


class Layer(object):
    def __init__(self, order: int, last_layer_dim: int, dim: int):
        self.order = order  # 层的序号, 从输入层(第0层)开始计数
        self.last_layer_dim = last_layer_dim  # 上一层的节点个数
        self.dim = dim  # 当前层的节点个数

        # w 不能全部置为0, 这样会导致每个层的所有节点的计算都是相同的
        # 如果激活函数使用sigmoid或者tanh, 随机出来的 w 最好小一点，一般乘以0.01.
        # 如果w比较大, 通过激活函数计算出来的值会落到接近1的位置, 导致学习速度变慢.
        # randn函数返回一个或一组样本，具有标准正态分布，又称为u分布，是以0为均值、以1为标准差的正态分布，记为N(0, 1)
        # 这里除以了sqrt(last_layer_dim), 用以减小w
        W = np.random.randn(last_layer_dim, dim) / np.sqrt(last_layer_dim)
        b = np.zeros((1, dim))

        self.weight = W  # 权重矩阵
        self.bias = b  # 偏置
        print(f'Layer: {order}, last_layer_dim={last_layer_dim}, dim={dim}')
        print(f'W :\n {W}')
        print(f'b :\n {b}')


class NeuralNetwork(object):
    def __init__(self):
        pass

    def build_model(self, X: np.ndarray, y: np.ndarray,  num_passes=20000, print_loss=False):
        '''
        Parameters
        ----------
        X : n * m(n列, m行)的矩阵, 其中n是特征向量的维度, m是数据集的样本个数, 每一行是一组特征向量
            例如, X是一个3*2的矩阵 [[0, 0, 1], [1, 1, 1]]
        y : 1 * m(1列, m行)的矩阵, m是数据集的样本个数, 每行的是输出值
            例如, y是一个1*2的矩阵 [[0], [1]]
        num_passes : 用梯度下降法训练数据的次数
        print_loss : 如果是 True, 每1000次打印一次误差

        Returns
        -------
        model :
        '''

        # 为随机数设置相同的种子, 这样, 每次开始训练时，得到的权重初始集的分布都是完全一致的.
        # 这便于观察策略变动是如何影响网络训练的。
        np.random.seed(1)

        # ndarray.shape -> (r1, r2) 返回值中， r1表示数组元素的个数, r2表示元素(下一级数组)中的元素个数, 以此类推
        # n 是特征向量的维度, m 是数据集的样本个数
        num_examples, input_dim = X.shape

        layer_dims = [3, 2]  # 各层的节点数, 不包括输入层

        print('start build model')
        print(f'input_dim={input_dim}, num_examples={num_examples}, num_passes={num_passes}')

        layer1 = Layer(1, input_dim, layer_dims[0])

        pass
        print(print_loss)
        print(np.sqrt(2))

        model = []
        return model


def main():
    import matplotlib.pyplot as plt
    import sklearn
    import sklearn.datasets
    import sklearn.linear_model
    import matplotlib

    # 调整图片的默认大小
    matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

    # 生成随机坐标点, 共两类，分别落在相互交叉的半圆里
    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    # plt.title("examples")
    # print(X)
    print('1-----------------------------')
    # print(y)
    print('2-----------------------------')
    # print(X[:, 0])
    print('3-----------------------------')
    # print(X[:, 1])

    neural_network = NeuralNetwork()
    model = neural_network.build_model(X, y, print_loss=True)


if __name__ == '__main__':
    main()
