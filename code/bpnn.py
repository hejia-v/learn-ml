# -*- coding:utf-8 -*-
import time
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y


def visualize(X, y, model):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = predict(model, np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()
    plt.title("bp nn")


def predict(model, x):
    layer1, layer2 = model
    feedforward(x, layer1)
    feedforward(layer1.a, layer2, True)
    return np.argmax(layer2.a, axis=1)


class Layer(object):
    def __init__(self, last_layer_dim: int, dim: int):
        self.W = np.random.randn(last_layer_dim, dim) / np.sqrt(last_layer_dim)
        self.b = np.zeros((1, dim))
        self.z = None
        self.a = None
        self.delta = None
        self.dW = None
        self.db = None


def softmax(X):  # m行dim列
    exp_scores = np.exp(X)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs

def feedforward(x: np.ndarray, layer: Layer, is_final: bool=False):
    layer.z = x.dot(layer.W) + layer.b  # m行dim列
    if is_final:
        layer.a = softmax(layer.z)
    else:
        layer.a = np.tanh(layer.z)


def backprop(x: np.ndarray, y: np.ndarray, layer: Layer, nextlayer: Layer,
            num_examples: int, is_final: bool=False):
    if is_final:
        delta = layer.a
        delta[range(num_examples), y] -= 1 # m行dim列, a_{[l]}^{(i)} - 1{y_{(i)}=l}
        # layer.dW = (x.T).dot(delta) / num_examples
        # layer.db = np.sum(delta, axis=0, keepdims=True) / num_examples
        layer.dW = (x.T).dot(delta)
        layer.db = np.sum(delta, axis=0, keepdims=True)
        layer.delta = delta
    else:
        delta = nextlayer.delta.dot(nextlayer.W.T) * (1 - np.power(layer.a, 2))  # * 对应元素相乘
        # layer.dW = np.dot(x.T, delta) / num_examples
        # layer.db = np.sum(delta, axis=0, keepdims=True) / num_examples
        layer.dW = np.dot(x.T, delta)
        layer.db = np.sum(delta, axis=0, keepdims=True)
        layer.delta = delta
    learn_rate = 0.01
    layer.W += -learn_rate * layer.dW
    layer.b += -learn_rate * layer.db


def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    num_examples = len(X)
    np.random.seed(0)
    input_dim = X.shape[1]
    nn_output_dim = 2

    layer1 = Layer(input_dim, nn_hdim)
    layer2 = Layer(nn_hdim, nn_output_dim)

    for i in range(0, num_passes):
        feedforward(X, layer1)
        feedforward(layer1.a, layer2, True)

        backprop(layer1.a, y, layer2, None, num_examples, True)
        backprop(X, y, layer1, layer2, num_examples)

    model = [layer1, layer2]
    return model


def main():
    X, y = generate_data()
    start_time = time.time()
    model = build_model(X, y, 3)
    time_cost = time.time() - start_time
    summary = f'cost: {time_cost}'
    print(summary)
    visualize(X, y, model)


if __name__ == "__main__":
    main()
