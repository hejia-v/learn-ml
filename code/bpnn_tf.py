# -*- coding:utf-8 -*-
import time
from sklearn import datasets
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def add_layer(inputs, dim_in, dim_out, layer_n, is_output_layer=False, y=None):
    layer_name = f'layer{layer_n}'

    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([dim_in, dim_out]))  # Weight中都是随机变量
            tf.summary.histogram(layer_name + "/weights", Weights)  # 可视化观看变量
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, dim_out]))  # biases推荐初始值不为0
            tf.summary.histogram(layer_name + "/biases", biases)  # 可视化观看变量
        with tf.name_scope('z'):
            z = tf.matmul(inputs, Weights) + biases  # inputs*Weight+biases
            tf.summary.histogram(layer_name + "/z", z)  # 可视化观看变量

        if is_output_layer:
            outputs = tf.nn.softmax(z, name='outputs')
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y, name='loss')
            tf.summary.histogram(layer_name + '/loss', loss)  # 可视化观看变量
            return outputs, loss
        else:
            outputs = tf.nn.tanh(z)
            tf.summary.histogram(layer_name + "/outputs", outputs)  # 可视化观看变量
            return outputs, None


np.random.seed(0)
X_data, y_data = datasets.make_moons(200, noise=0.20)
num_examples = len(X_data)
ym_data = np.zeros((num_examples, 2))
ym_data[range(num_examples), y_data] = 1


# 生成一个带可展开符号的域
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, name='X')
    ys = tf.placeholder(tf.float32, name='y')

tf.set_random_seed(0)
# 三层神经网络，输入层（2个神经元），隐藏层（3神经元），输出层（2个神经元）
layer1, _ = add_layer(xs, 2, 3, 1)  # 隐藏层
predict_step, loss = add_layer(layer1, 3, 2, 2, True, ys)  # 输出层

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # 0.01学习率,minimize(loss)减小loss误差

init = tf.global_variables_initializer()

config = tf.ConfigProto()
# https://tensorflow.google.cn/tutorials/using_gpu#allowing_gpu_memory_growth
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# 合并到Summary中
merged = tf.summary.merge_all()

# 选定可视化存储目录
writer = tf.summary.FileWriter("./", sess.graph)

sess.run(init)  # 先执行init

start_time = time.time()
# 训练2w次
num_passes = 20000
for i in range(num_passes):
    sess.run(train_step, feed_dict={xs: X_data, ys: ym_data})
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={xs: X_data, ys: ym_data})  # merged也是需要run的
        writer.add_summary(result, i)  # result是summary类型的，需要放入writer中，i步数（x轴）
time_cost = time.time() - start_time
summary_text = f'cost: {time_cost}'
print(summary_text)

# --------------------------- predict ---------------------------
def predict(x):
    predict = sess.run(predict_step, feed_dict={xs: x})
    return np.argmax(predict, axis=1)


def visualize(X, y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()
    plt.title("bp nn")

visualize(X_data, y_data)
