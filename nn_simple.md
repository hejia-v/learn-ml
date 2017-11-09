# 一个简单的神经网络的实现

不详述神经网络模型，只记录一下实现神经网络时的推导过程。

## 0. 数学复习

矩阵A(m, n)，m指行数，n指列数

## 1. 输入值和输出值

输入值是一个维度是n的特征向量，记作 $x=(x_1,x_2,\ldots,x_n)$。一个数据集里一般有多个样本，假定有m个样本，则将这数据集记作 $X = (x^{(1)},x^{(2)},\ldots,x^{(m)})$ 。

输出值一般是一个数值，用于表示属于哪个类别。对于m个样本，输出值的集合记作 $y=(y^{(1)}, y^{(2)}, \ldots, y^{(m)})$ 。

输入层的节点个数取决于输入的特征向量的维度。

输出层的节点个数取决于拥有的类别个数。如果只有2类，则可以只用一个输出节点用于预测0或1。

隐藏层的节点越多，则越复杂的函数可以fit到。但是这样做的代价也比较大，首先，会增大训练参数和进行预测的计算量。其次，大量的参数也容易导致过拟合。  需要根据具体情况选择节点的个数。

## 2. 权重和偏置的初始化

## 3. 正向传播

神经网络使用正向传播进行预测。

这里约定上标 (i) 为样本在样本集中的序号，上标 [i] 为神经网络的序号，下标 [i] 为某一隐藏层的节点的序号。

对于一个3层的神经网络，可以这样计算预测值 $\hat y$ ：
$$
\begin{align}
& z^{[1]} = W^{[1]}x+ b^{[1]} \\
& a^{[1]} = \sigma (z^{[1]}) \\
& z^{[2]} = W^{[2]}a^{[1]} + b^{[2]} \\
& a^{[2]} = \hat y = softmax (z^{[2]}) \\
\end{align}
$$
神经网络里的计算都需要进行向量化，具体来说

对于第1层的输入值和输出值
$$
X =   \begin{pmatrix}
        x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\
        x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\
        \vdots     & \vdots    & \ddots & \vdots \\
        x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)} \\
        \end{pmatrix} = \begin{pmatrix}
        x^{(1)} \\
        x^{(2)} \\
        \vdots \\
        x^{(m)} \\
        \end{pmatrix} , \qquad y = \begin{pmatrix}
        y^{(1)} \\
        y^{(2)} \\
        \vdots \\
        y^{(m)} \\
        \end{pmatrix}
$$
第2层

假设第2层有u个节点
$$
W^{[1]} = \begin{pmatrix}
        w_{1[1]} & w_{1[2]} & \cdots & w_{1[u]} \\
        w_{2[1]} & w_{2[2]} & \cdots & w_{2[u]} \\
        \vdots     & \vdots    & \ddots & \vdots \\
        w_{n[1]} & w_{n[2]} & \cdots & w_{n[u]} \\
        \end{pmatrix} = \begin{pmatrix}
        w_{[1]} & w_{[2]} & \ldots & w_{[u]} \\
        \end{pmatrix} , \qquad b^{[1]} = \begin{pmatrix}
        b_{[1]} \\
        b_{[2]} \\
        \vdots \\
        b_{[u]} \\
        \end{pmatrix}
$$
根据矩阵的分块的性质
$$
\begin{align}
W^{[1]}X^T &= \begin{pmatrix}
        w_{[1]} \\ w_{[2]} \\ \vdots \\ w_{[u]} \\
        \end{pmatrix} \begin{pmatrix}
        x^{(1)} & x^{(2)} & \ldots & x^{(m)} \\
        \end{pmatrix} = AB = \begin{pmatrix}
        A_{11} \\ A_{21} \\ \vdots \\ A_{u1} \\
        \end{pmatrix}  \begin{pmatrix}
        B_{11} & B_{12} & \ldots & B_{1m} \\
        \end{pmatrix} \\
       & = \begin{pmatrix}
        C_{11} & C_{12} & \cdots & C_{1m} \\
        C_{21} & C_{22} & \cdots & C_{2m} \\
        \vdots     & \vdots    & \ddots & \vdots \\
        C_{u1} & C_{u2} & \cdots & C_{um} \\
        \end{pmatrix} = \begin{pmatrix}
        A_{11}B_{11} & A_{11}B_{12} & \cdots & A_{11}B_{1m} \\
        A_{21}B_{11} & A_{21}B_{12} & \cdots & A_{21}B_{1m} \\
        \vdots     & \vdots    & \ddots & \vdots \\
        A_{u1}B_{11} & A_{u1}B_{12} & \cdots & A_{u1}B_{1m} \\
        \end{pmatrix}  \\
        & =  \begin{pmatrix}
        w_{[1]}x^{(1)} & w_{[1]}x^{(2)} & \cdots & w_{[1]}x^{(m)} \\
        w_{[2]}x^{(1)} & w_{[2]}x^{(2)} & \cdots & w_{[2]}x^{(m)} \\
        \vdots     & \vdots    & \ddots & \vdots \\
        w_{[u]}x^{(1)} & w_{[u]}x^{(2)} & \cdots & w_{[u]}x^{(m)} \\
        \end{pmatrix}
\end{align}
$$
参考numpy中的boardcast，则有
$$
\begin{align}
z^{[1]} &= W^{[1]}X^T + boardcast\ (b^{[1]}) =  \begin{pmatrix}
        w_{[1]}x^{(1)} + b_{[1]} & w_{[1]}x^{(2)} + b_{[1]} & \cdots & w_{[1]}x^{(m)} + b_{[1]} \\
        w_{[2]}x^{(1)} + b_{[2]} & w_{[2]}x^{(2)} + b_{[2]} & \cdots & w_{[2]}x^{(m)} + b_{[2]} \\
        \vdots     & \vdots    & \ddots & \vdots \\
        w_{[u]}x^{(1)} + b_{[u]} & w_{[u]}x^{(2)} + b_{[u]} & \cdots & w_{[u]}x^{(m)} + b_{[u]} \\
        \end{pmatrix} \\
        & =  \begin{pmatrix}
        z_{[1]}^{(1)} & z_{[1]}^{(2)} & \cdots & z_{[1]}^{(m)} \\
        z_{[2]}^{(1)} & z_{[2]}^{(2)} & \cdots & z_{[2]}^{(m)} \\
        \vdots     & \vdots    & \ddots & \vdots \\
        z_{[u]}^{(1)} & z_{[u]}^{(2)} & \cdots & z_{[u]}^{(m)} \\
        \end{pmatrix}
\end{align}
$$

$$
a^{[1]} =  \begin{pmatrix}
        \sigma(z_{[1]}^{(1)}) & \sigma(z_{[1]}^{(2)}) & \cdots & \sigma(z_{[1]}^{(m)}) \\
        \sigma(z_{[2]}^{(1)}) & \sigma(z_{[2]}^{(2)}) & \cdots & \sigma(z_{[2]}^{(m)}) \\
        \vdots     & \vdots    & \ddots & \vdots \\
        \sigma(z_{[u]}^{(1)}) & \sigma(z_{[u]}^{(2)}) & \cdots & \sigma(z_{[u]}^{(m)}) \\
        \end{pmatrix}
$$

以上的计算方式可以推广到多层神经网络中的任一隐藏层。

Because we want our network to output probabilities the activation function for the output layer will be the [softmax](https://en.wikipedia.org/wiki/Softmax_function), which is simply a way to convert raw scores to probabilities. If you're familiar with the logistic function you can think of softmax as its generalization to multiple classes.

激活函数

推广到多层神经网络

#### 向量化

举一个直观的例子

由于手算比较耗时，这里使用mathematica进行符号计算，在mathematica中输入以下代码

```
{u, n, m} = {3, 2, 5}
Array[Subscript[w, ##] &, {u, n}] . Array[Subscript[x, ##] &, {n, m}]
```

得到结果
$$
\left(
\begin{array}{ccccc}
 w_{1,1} x_{1,1}+w_{1,2} x_{2,1} & w_{1,1} x_{1,2}+w_{1,2} x_{2,2} & w_{1,1} x_{1,3}+w_{1,2} x_{2,3} & w_{1,1} x_{1,4}+w_{1,2} x_{2,4} & w_{1,1} x_{1,5}+w_{1,2} x_{2,5} \\
 w_{2,1} x_{1,1}+w_{2,2} x_{2,1} & w_{2,1} x_{1,2}+w_{2,2} x_{2,2} & w_{2,1} x_{1,3}+w_{2,2} x_{2,3} & w_{2,1} x_{1,4}+w_{2,2} x_{2,4} & w_{2,1} x_{1,5}+w_{2,2} x_{2,5} \\
 w_{3,1} x_{1,1}+w_{3,2} x_{2,1} & w_{3,1} x_{1,2}+w_{3,2} x_{2,2} & w_{3,1} x_{1,3}+w_{3,2} x_{2,3} & w_{3,1} x_{1,4}+w_{3,2} x_{2,4} & w_{3,1} x_{1,5}+w_{3,2} x_{2,5} \\
\end{array}
\right)
$$

## 4. 反向传播

3、终止条件
偏重的更新低于某个阈值；
预测的错误率低于某个阈值；
达到预设一定的循环次数；

## 5. 交叉验证方法

