# 一个简单的BP神经网络的实现

不详述神经网络模型，只记录一下实现BP神经网络时的推导过程。

<!--more-->

## 输入值和输出值

输入值是一个维度是n的特征向量，记作 $x=(x_1,x_2,\ldots,x_n)$。一个数据集里一般有多个样本，假定有m个样本，则将这数据集记作 $X = (x^{(1)},x^{(2)},\ldots,x^{(m)})$ 。

输出值一般是一个数值，用于表示属于哪个类别。对于m个样本，输出值的集合记作 $y=(y^{(1)}, y^{(2)}, \ldots, y^{(m)})$ 。

输入层的节点个数取决于输入的特征向量的维度。

输出层的节点个数取决于拥有的类别个数。如果只有2类，则可以只用一个输出节点用于预测0或1。

隐藏层的节点越多，则越复杂的函数可以fit到。但是这样做的代价也比较大，首先，会增大训练参数和进行预测的计算量。其次，大量的参数也容易导致过拟合。  需要根据具体情况选择节点的个数。

有一个经验公式可以确定隐含层节点数目，如下
$$
h=\sqrt{m+n} + a
$$
其中h为隐含层节点数目，m为输入层节点数目，n为输出层节点数目，a为1~10之间的调节常数。

## 正向传播

> 这里约定上标 (i) 为样本在样本集中的序号，上标 [i] 为神经网络的层的序号，下标 [i] 为网络中某一层的节点的序号，log的底数默认是e。

神经网络使用正向传播进行预测。

对于一个3层的神经网络，可以这样计算预测值 $\hat y$ ：
$$
\begin{align}
& z^{[1]} = xW^{[1]}+ b^{[1]} \\
& a^{[1]} = \sigma (z^{[1]}) \\
& z^{[2]} = a^{[1]}W^{[2]} + b^{[2]} \\
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

假设第2层有u个节点，则这层的权重和偏置为
$$
W^{[1]} = \begin{pmatrix}
        w_{1[1]} & w_{1[2]} & \cdots & w_{1[u]} \\
        w_{2[1]} & w_{2[2]} & \cdots & w_{2[u]} \\
        \vdots     & \vdots    & \ddots & \vdots \\
        w_{n[1]} & w_{n[2]} & \cdots & w_{n[u]} \\
        \end{pmatrix} 
        = \begin{pmatrix}
        w_{[1]} & w_{[2]} &  \ldots & w_{[u]} 
        \end{pmatrix} , \qquad 
b^{[1]} = \begin{pmatrix}
        b_{[1]} & b_{[2]} & \ldots & b_{[u]} \end{pmatrix}
$$
根据矩阵的分块的性质
$$
\begin{align}
XW^{[1]} &= \begin{pmatrix}
        x^{(1)} \\ x^{(2)} \\ \vdots \\ x^{(m)} \\
        \end{pmatrix}
	\begin{pmatrix}
        w_{[1]} & w_{[2]} &  \ldots & w_{[u]} 
        \end{pmatrix} = 
AB = \begin{pmatrix}
        A_{11} \\ A_{21} \\ \vdots \\ A_{m1} \\
        \end{pmatrix}  \begin{pmatrix}
        B_{11} & B_{12} & \ldots & B_{1u} \\
        \end{pmatrix} \\
       & = \begin{pmatrix}
        C_{11} & C_{12} & \cdots & C_{1u} \\
        C_{21} & C_{22} & \cdots & C_{2u} \\
        \vdots     & \vdots    & \ddots & \vdots \\
        C_{m1} & C_{m2} & \cdots & C_{mu} \\
        \end{pmatrix} 
        = \begin{pmatrix}
        A_{11}B_{11} & A_{11}B_{12} & \cdots & A_{11}B_{1u} \\
        A_{21}B_{11} & A_{21}B_{12} & \cdots & A_{21}B_{1u} \\
        \vdots     & \vdots    & \ddots & \vdots \\
        A_{m1}B_{11} & A_{m1}B_{12} & \cdots & A_{m1}B_{1u} \\
        \end{pmatrix}  \\
        & =  \begin{pmatrix}
        x^{(1)}w_{[1]} & x^{(1)}w_{[2]} & \cdots & x^{(1)}w_{[u]} \\
        x^{(2)}w_{[1]} & x^{(2)}w_{[2]} & \cdots & x^{(2)}w_{[u]} \\
        \vdots     & \vdots    & \ddots & \vdots \\
        x^{(m)}w_{[1]} & x^{(m)}w_{[2]} & \cdots & x^{(m)}w_{[u]} \\
        \end{pmatrix}
\end{align}
$$
参考numpy中的boardcast，则有
$$
\begin{align}
z^{[1]} &= XW^{[1]} + boardcast\ (b^{[1]}) =  \begin{pmatrix}
        x^{(1)}w_{[1]} + b_{[1]} & x^{(1)}w_{[2]} + b_{[2]} & \cdots & x^{(1)}w_{[u]} + b_{[u]} \\
        x^{(2)}w_{[1]} + b_{[1]} & x^{(2)}w_{[2]} + b_{[2]} & \cdots & x^{(2)}w_{[u]} + b_{[u]} \\
        \vdots     & \vdots    & \ddots & \vdots \\
        x^{(m)}w_{[1]} + b_{[1]} & x^{(m)}w_{[2]} + b_{[2]} & \cdots & x^{(m)}w_{[u]} + b_{[u]} \\
        \end{pmatrix} \\
        & =  \begin{pmatrix}
        z_{[1]}^{(1)} & z_{[2]}^{(1)} & \cdots & z_{[u]}^{(1)} \\
        z_{[1]}^{(2)} & z_{[2]}^{(2)} & \cdots & z_{[u]}^{(2)} \\
        \vdots     & \vdots    & \ddots & \vdots \\
        z_{[1]}^{(m)} & z_{[2]}^{(m)} & \cdots & z_{[u]}^{(m)} \\
        \end{pmatrix}
\end{align}
$$

$$
a^{[1]} =  \begin{pmatrix}
        \sigma(z_{[1]}^{(1)}) & \sigma(z_{[2]}^{(1)}) & \cdots & \sigma(z_{[u]}^{(1)}) \\
        \sigma(z_{[1]}^{(2)}) & \sigma(z_{[2]}^{(2)}) & \cdots & \sigma(z_{[u]}^{(2)}) \\
        \vdots     & \vdots    & \ddots & \vdots \\
        \sigma(z_{[1]}^{(m)}) & \sigma(z_{[2]}^{(m)}) & \cdots & \sigma(z_{[u]}^{(m)}) \\
        \end{pmatrix}
$$

以上的计算方式可以推广到多层神经网络中的任一隐藏层，对于第$l$层，将 $X$ 类比成 $a^{[l-1]}$ ， $W^{[1]}$ 类比成 $W^{[l]}$ ， $b^{[1]}$ 类比成 $b^{[l]}$ ， $z^{[1]}$ 类比成 $z^{[l]}$ ， $a^{[1]}$ 类比成 $a^{[l]}$ ，就可以了。

对于输出层，因为我们希望网络输出各个分类的概率，所以将输出层的激活函数选为[softmax](https://en.wikipedia.org/wiki/Softmax_function) 。softmax 是一种简便的将原始评分转换成概率的方法。可以将 softmax 看做 logistic 函数的在多分类下的推广。
$$
a^{[2]} =  softmax (z^{[2]}) = \begin{pmatrix}
        softmax(z^{[2](1)}) \\ softmax(z^{[2](2)}) \\ \vdots \\ softmax(z^{[2](m)}) \\
        \end{pmatrix}
$$


### 权重和偏置的初始化

权重 w 不能全部置为0，这样会导致每个层的所有节点的计算都是相同的。如果激活函数使用sigmoid或者tanh，随机出来的 w 最好小一点，一般乘以0.01。因为如果 w 比较大，通过激活函数计算出来的值会落到接近1的位置，导致学习速度变慢。

偏置 b 可以全部初始化为0，但是推荐初始值不为0。

### 激活函数

- [tanh](https://en.wikipedia.org/wiki/Hyperbolic_function#Tanh)
- [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function)
- <a href="https://en.wikipedia.org/wiki/Rectifier_(neural_networks)">ReLUs</a>

### softmax

[softmax](https://en.wikipedia.org/wiki/Softmax_function) 又称为归一化指数函数，是广义上的 logistic 函数。假设k 维向量 $z$ 的各项是任意实数，使用 softmax 对 $z$ 进行“压缩”处理后，压缩后的向量 $\sigma (z)$ 的每一项的值在 [0, 1] 之间，所有项之和等于1。函数公式如下
$$
\sigma (z)_j = \frac {e^{z_j}}{ \sum_{k=1}^K e^{z_k} } \quad  for\ j = 1, \ldots, K.
$$

一个简单的例子如下

```python
import numpy as np
z = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
exp_scores = np.exp(z)
probs = exp_scores / np.sum(exp_scores)  # 应用 softmax
```

在神经网络的输出层中，有C个分类，对于给定的输入 $z$ ，每个分类的概率可以表示为：
$$
\begin{bmatrix}
    P(t=1|z) \\ P(t=2|z) \\ \vdots \\ P(t=C|z) \\
\end{bmatrix} = \frac {1}{ \sum_{k=1}^C e^{z_k} } \begin{bmatrix}
    e^{z_1} \\ e^{z_2} \\ \vdots \\ e^{z_C} \\
\end{bmatrix}
$$
其中，$P(t=c|z)$ 表示，在给定输入$z$时，该输入数据是$c$分类的概率。

对于 softmax 函数
$$
a_j = \frac {e^{z_j}}{ \sum_{k=1}^K e^{z_k} } \quad  for\ j = 1, \ldots, K.
$$
softmax 函数的求导过程比较特别，分如下2种情况。这是因为i = j 时，$z_i$ 与 $z_j$ 是同一个变量，按偏导数的定义，将多元函数关于一个自变量求偏导数时，就将其余的自变量看成常数，因此需要分2种情况处理。
$$
\begin{align}
&if \ j=i \\
&\qquad \frac{ \partial a_j }{ \partial z_i } = \frac{ \partial }{ \partial z_i } \Bigl( \frac {e^{z_j}}{ \sum_{k=1}^K e^{z_k} } \Bigr)
= \frac { (e^{z_j})^\prime \cdot \sum_k e^{z_k} - e^{z_j} \cdot e^{z_j} } { \Bigl( \sum_k e^{z_k} \Bigr)^2 }
= \frac { e^{z_j} } { \sum_k e^{z_k} } - \frac { e^{z_j} } { \sum_k e^{z_k} } \cdot \frac { e^{z_j} } { \sum_k e^{z_k} }
= a_j ( 1-a_j )
\\
&if \ j \neq i \\
&\qquad \frac{ \partial a_j }{ \partial z_i } = \frac{ \partial }{ \partial z_i } \Bigl( \frac {e^{z_j}}{ \sum_{k=1}^K e^{z_k} } \Bigr)
= \frac { 0 \cdot \sum_k e^{z_k} - e^{z_j} \cdot e^{z_i} } { \Bigl( \sum_k e^{z_k} \Bigr)^2 }
= - \frac { e^{z_j} } { \sum_k e^{z_k} } \cdot \frac { e^{z_i} } { \sum_k e^{z_k} }
= -a_j a_i
\end{align}
$$

## 反向传播

反向传播主要思想是：
（1）将训练集数据输入到输入层，经过隐藏层，最后达到输出层并输出结果，这是前向传播过程；
（2）由于输出层的输出结果与实际结果有误差，则计算估计值与实际值之间的误差，并将该误差从输出层向隐藏层反向传播，直至传播到输入层；
（3）在反向传播的过程中，根据误差调整各种参数的值；不断迭代上述过程，直至收敛。

### 代价函数

我们把定义估计值与实际值之间误差(单个样本)的函数叫作误差损失(loss)函数，代价(cost)函数是各个样本的loss函数的平均。

如果误差损失函数采用二次代价函数 ，在实际中，如果误差越大，参数调整的幅度可能更小，训练更缓慢。使用交叉熵代价函数替换二次代价函数，可以解决学习缓慢的问题。

> 示性函数：$1\{\cdot\}$
> 取值规则为：$1\{值为真的表达式\}=1$ ，$1\{值为假的表达式\}=0$ 。举例来说，表达式 $1\{2+2=4\}$的值为1，$1\{1+1=5\}$的值为 0。

在分类问题中，交叉熵代价函数与对数似然代价函数在形式上是基本一致的。

对于输出层的softmax激活函数，假设有m个样本，k个类别，将$p(x)$记为$1\{ y^{(i)} = j \}$，$q(x)$记为softmax函数的输出值

则根据交叉熵公式，
$$
H(p,q) = - \sum_x p(x)log\ q(x)
$$
可以得到交叉熵代价函数为：
$$
J(\theta) = -\frac{1}{m} \Biggl[ \sum_{i=1}^m  \biggl[ \sum_{j=1}^k 1\{ y^{(i)} = j \} \ log \frac{e^{\theta_j^T x^{(i)}}}{ \sum_{l=1}^k e^{ \theta_l^T x^{(i)} } }  \biggl] \Biggl]
$$
从似然函数的角度分析，记$h_{\theta j}(x)= \frac{e^{\theta_j^T x}}{ \sum_{l=1}^k e^{ \theta_l^T x } }$  (h一般是hypothesis的缩写)，在一个样本中，对于输入x的分类结果为j的概率为
$$
P(y=j|x; \theta) = h_{\theta j} (x)^{ 1\{ y = j \}  }
$$
将所有分类的概率综合起来，则有：
$$
P(y|x; \theta) = \prod_{j=1}^k h_{\theta j} (x)^{ 1\{ y = j \}  }
$$
取似然函数为：
$$
\begin{align}
L(\theta) &= \prod_{i=1}^m P(y^{(i)} | x^{(i)}; \theta) \\
&= \prod_{i=1}^m \biggl[ \prod_{j=1}^k h_{\theta j} (x^{(i)})^{1\{ y^{(i)} = j \}} \biggl]
\end{align}
$$
对数似然函数为：
$$
\begin{align}
l(\theta) &= log\ L(\theta) \\
&= \sum_{i=1}^m \biggl[ \sum_{j=1}^k 1\{ y^{(i)} = j \} \ log\ h_{\theta j} (x^{(i)})  \biggl]
\end{align}
$$

最大似然估计就是要求得使$l(\theta)$取最大值时的$\theta$ 。一般将它乘上一个负系数**-1/m**，即：
$$
J(\theta) = - \frac{1}{m} l(\theta)
$$
则$J(\theta)$取最小值时的$\theta$为要求的最佳参数。这也就是上面的交叉熵代价函数。


> 1. log MN = log M + log N
> 2. 很多文献里对数都没标底数，这说明可以取任意底数，一般取e或2，取不同的底数时，对数值只相差了一个常数系数，对算法不会有影响。

相关的具体分析参考如下链接：
- [Softmax回归](http://deeplearning.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92)
- [Softmax Regression](http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/)
- [Softmax分类函数](http://www.jianshu.com/p/8eb17fa41164)
- [neural_network_implementation_intermezzo02](https://github.com/peterroelants/peterroelants.github.io/blob/master/notebooks/neural_net_implementation/neural_network_implementation_intermezzo02.ipynb)
- [Improving the way neural networks learn](http://neuralnetworksanddeeplearning.com/chap3.html)
- [交叉熵代价函数](http://blog.csdn.net/u014313009/article/details/51043064)


### 交叉熵代价函数 [cross-entropy loss](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression)

#### 二次代价函数的不足

考察一下二次代价函数
$$
C = \frac {1}{2n} \sum_x \| y(x) - a^L(x) \|^2
$$
其中，C表示代价，x表示样本，y表示实际值，a表示输出值，n表示样本的总数。为简单起见，以一个样本为例进行说明，此时二次代价函数为：
$$
C = \frac {(y-a)^2}{2}
$$
在用梯度下降法(gradient descent) 调整权重w和偏置b的过程中，w和b的梯度推导如下：
$$
\begin{align}
&\frac { \partial C }{ \partial w } = (a - y)\sigma^\prime(z) x \\
&\frac { \partial C }{ \partial b } = (a - y)\sigma^\prime(z) \\
\end{align}
$$
可以看出，w和b的梯度跟激活函数的梯度成正比，激活函数的梯度越大，w和b的大小调整得越快，训练收敛得就越快。而神经网络常用的激活函数为sigmoid函数或tanh函数，观察这些激活函数的图像，当初始的代价（误差）越大时，梯度（导数）越小，训练的速度越慢。这与我们的期望不符，即：不能像人一样，错误越大，改正的幅度越大，从而学习得越快。

#### 交叉熵代价函数

为了克服二次代价函数学习缓慢的缺点，引入了交叉熵代价函数：
$$
C = - \frac{1}{n} \sum_x [yln\ a + (1-y)ln(1-a)]
$$

其中，x表示样本，n表示样本的总数。重新计算参数w的梯度：
$$
\begin{align}
\frac{\partial C}{\partial w_j} &= -\frac{1}{n} \sum_x \Bigl( \frac{y}{\sigma(z)} -\frac{(1-y)}{1-\sigma(z)} \Bigr) \frac{\partial \sigma(z)}{\partial w_j} \\
&= -\frac{1}{n} \sum_x \Bigl( \frac{y}{\sigma(z)} -\frac{(1-y)}{1-\sigma(z)} \Bigr) \sigma^\prime(z)x_j \\
&= \frac{1}{n} \sum_x  \frac{ \sigma^\prime(z)x_j }{\sigma(z) (1-\sigma(z))} (\sigma(z)-y) \\
&= \frac{1}{n} \sum_x  x_j (\sigma(z)-y) \\
\end{align}
$$
其中，w的梯度公式中原来的 $\sigma^\prime(z)$ 被消掉了；另外，该梯度公式中的 $\sigma(z)-y$ 表示输出值与实际值之间的误差。所以，当误差越大，梯度就越大，参数w调整得越快，训练速度也就越快。同理可得，b的梯度为：
$$
\frac{\partial C}{\partial b} = \frac{1}{n} \sum_x (\sigma(z)-y)
$$

#### 交叉熵代价函数的来源

用交叉熵代替二次代价函数的想法源自哪里？

以偏置b的梯度计算为例
$$
\begin{align}
\frac{\partial C}{\partial b} &= \frac{\partial C}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial b} \\
&= \frac{\partial C}{\partial a} \cdot \sigma^\prime(z) \cdot \frac{\partial (wx+b)}{\partial b} \\
&= \frac{\partial C}{\partial a} \cdot \sigma^\prime(z) \\
&= \frac{\partial C}{\partial a} \cdot a(1-a) \\
\end{align}
$$
而二次代价函数推导出来的b的梯度公式为：
$$
\frac { \partial C }{ \partial b } = (a - y)\sigma^\prime(z)
$$
为了消掉该公式中的 $\sigma^\prime(z)$ ，需要找到一个代价函数使得：
$$
\frac { \partial C }{ \partial b } = (a - y)
$$
即
$$
\frac { \partial C }{ \partial a } \cdot a(1-a) = (a - y)
$$
对方程进行关于a的积分，可得：
$$
C = -[yln\ a + (1-y)ln(1-a)] + constant
$$
其中constant是积分常量。这是一个单独训练样本X对代价函数的贡献。为了得到整个的代价函数，还需要对所有的训练样本进行平均，可得：
$$
C = -\frac{1}{n} \sum_x [yln\ a + (1-y)ln(1-a)] + constant
$$
而这就是前面的交叉熵代价函数。

在分类问题中，交叉熵其实就是对数似然函数的最大化。

关于交叉熵的更多内容，参考以下链接：
- [交叉熵（Cross-Entropy）](http://blog.csdn.net/rtygbwwwerr/article/details/50778098)
- [Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy)
- [怎样理解 Cross Entropy](http://shuokay.com/2017/06/23/cross-entropy/)
- [信息论的熵](http://blog.csdn.net/hguisu/article/details/27305435)
- <a href="https://en.wikipedia.org/wiki/Entropy_(information_theory)">Entropy</a>

### 反向传播算法

> 求导的链式法则 ([Chain rule](https://en.wikipedia.org/wiki/Chain_rule))：
> 表达式：$(f(g(x)))^\prime = f^\prime (g(x)) g^\prime (x)$
> 其他形式：$\frac {dy}{dx} = \frac {dy}{dz} \cdot \frac {dz}{dx}$

求$J(W, b)$的最小值可以使用梯度下降法，根据梯度下降法可得$W$和$b$的更新过程：
$$
W^{[l]}_{[i]j} := W^{[l]}_{[i]j} - \alpha \frac{\partial}{\partial \ W^{[l]}_{[i]j}} J(W, b) \\
b^{[l]}_{[i]} := b^{[l]}_{[i]} - \alpha \frac{\partial}{\partial \ b^{[l]}_{[i]}} J(W, b)
$$
其中，$\alpha$为学习步长，$W^{[l]}_{[i]j}$为第l层的第i个节点的权重第j个分量，$b^{[l]}_{[i]}$为第l层的第i个节点的偏置。

#### 输出层

输出层的激活函数采用的是softmax函数。根据前文，输出层的误差采用交叉熵代价函数来衡量，即：
$$
x = a ^{[1]}\\
z_{[j]}^{[2]} = xW_{[j]}^{[2]} + b_{[j]}^{[2]}  \\
a_{[j]}^{[2]} = \frac{e^{z_{[j]}^{[2]}}}{ \sum_{k=1}^K e^{ z_{[k]}^{[2]} } }   \\
J(W, b) = -\frac{1}{m} \Biggl[ \sum_{i=1}^m  \biggl[ \sum_{j=1}^k 1\{ y^{(i)} = j \} \ log \ a_{[j]}^{[2]}  \biggl] \Biggl]
$$
其中，依照前文约定，下标 [j] 为第$j$个节点的序号，上标 (i) 为样本在样本集中的序号。

输出层的第$l$个节点的权重$W_{[l]}$的第$t$个分量，求导(梯度)为：(为了简化公式，没有加上标$[2]$，a和z没有加上标$(i)$)
$$
\begin{align}
\frac{\partial}{\partial \ W_{[l]t}} J(W, b) &= -\frac{1}{m} \Biggl[ \sum_{i=1}^m \biggl[ \frac{\partial}{\partial \ W_{[l]t}} \biggl[ \sum_{j=1}^k 1\{ y^{(i)} = j \} \ log\ a_{[j]}  \biggl] \biggl] \Biggl] \\
&= -\frac{1}{m} \Biggl[ \sum_{i=1}^m \biggl[ \frac{\partial}{\partial \ z_{[l]}} \biggl[ \sum_{j=1}^k 1\{ y^{(i)} = j \} \ log\ a_{[j]}  \biggl] \biggl] \cdot \frac{\partial \ z_{[l]}}{\partial \ W_{[l]t}} \Biggl] \\
&= -\frac{1}{m} \Biggl[ \sum_{i=1}^m \biggl[ \frac{\partial}{\partial \ z_{[l]}} \biggl[ \sum_{j=1}^k 1\{ y^{(i)} = j \} \ log\ a_{[j]}  \biggl] \biggl] \cdot x^{(i)}_t \Biggl] \\
&= -\frac{1}{m} \Biggl[ \sum_{i=1}^m \biggl[ \frac{\partial}{\partial \ z_{[l]}} \biggl( 1\{ y^{(i)} = l \} \ log\ a_{[l]}  \biggl) +  \frac{\partial}{\partial \ z_{[l]}} \biggl( \sum_{j=1,j \neq l}^k 1\{ y^{(i)} = j \} \ log\ a_{[j]} \biggr) \biggl] \cdot x^{(i)}_t \Biggl] \\
&= -\frac{1}{m} \Biggl[ \sum_{i=1}^m \biggl[ 1\{ y^{(i)} = l \} \biggl( \frac{\partial \ (log\ a_{[l]}) }{\partial \ a_{[l]}} \cdot \frac{\partial \ a_{[l]}}{\partial \ z_{[l]}} \biggl) +  \biggl( \sum_{j=1,j \neq l}^k 1\{ y^{(i)} = j \} \cdot \frac{\partial \ (log\ a_{[j]}) }{\partial \ a_{[j]}} \cdot \frac{\partial \ a_{[j]}}{\partial \ z_{[l]}} \biggr)  \biggl] \cdot x^{(i)}_t \Biggl] \\
&= -\frac{1}{m} \Biggl[ \sum_{i=1}^m \biggl[ 1\{ y^{(i)} = l \} \biggl( \frac{1}{a_{[l]}} \cdot a_{[l]} ( 1-a_{[l]} ) \biggl) +  \biggl( \sum_{j=1,j \neq l}^k 1\{ y^{(i)} = j \} \cdot \frac{1}{a_{[j]}} \cdot (-a_{[j]} a_{[l]}) \biggr)  \biggl] \cdot x^{(i)}_t \Biggl] \\
&= -\frac{1}{m} \Biggl[ \sum_{i=1}^m \biggl[ 1\{ y^{(i)} = l \} \cdot ( 1-a_{[l]} ) - \sum_{j=1,j \neq l}^k 1\{ y^{(i)} = j \} \cdot a_{[l]} \biggl] \cdot x^{(i)}_t \Biggl] \\
&= -\frac{1}{m} \Biggl[ \sum_{i=1}^m \biggl[ 1\{ y^{(i)} = l \} - 1\{ y^{(i)} = l \} \cdot a_{[l]} - \sum_{j=1,j \neq l}^k 1\{ y^{(i)} = j \} \cdot a_{[l]} \biggl] \cdot x^{(i)}_t \Biggl] \\
&= -\frac{1}{m} \Biggl[ \sum_{i=1}^m \biggl[ 1\{ y^{(i)} = l \} - a_{[l]} \cdot \sum_{j=1}^k 1\{ y^{(i)} = j \} \biggl] \cdot x^{(i)}_t \Biggl] \\
&= -\frac{1}{m} \Biggl[ \sum_{i=1}^m \biggl( 1\{ y^{(i)} = l \} - a_{[l]} \biggl) \cdot x^{(i)}_t \Biggl] \\
\end{align}
$$
输出层的第$l$个节点的偏置$b_{[l]}$的求导(梯度)为：(为了简化公式，没有加上标$[2]$，a和z没有加上标$(i)$)
$$
\begin{align}
\frac{\partial}{\partial \ b_{[l]}} J(W, b) &= -\frac{1}{m} \Biggl[ \sum_{i=1}^m \biggl[ \frac{\partial}{\partial \ b_{[l]}} \biggl[ \sum_{j=1}^k 1\{ y^{(i)} = j \} \ log\ a_{[j]}  \biggl] \biggl] \Biggl] \\
&= -\frac{1}{m} \Biggl[ \sum_{i=1}^m \biggl[ \frac{\partial}{\partial \ z_{[l]}} \biggl[ \sum_{j=1}^k 1\{ y^{(i)} = j \} \ log\ a_{[j]}  \biggl] \biggl] \cdot \frac{\partial \ z_{[l]}}{\partial \ b_{[l]}} \Biggl] \\
&= -\frac{1}{m} \Biggl[ \sum_{i=1}^m \biggl[ \frac{\partial}{\partial \ z_{[l]}} \biggl( 1\{ y^{(i)} = l \} \ log\ a_{[l]}  \biggl) +  \frac{\partial}{\partial \ z_{[l]}} \biggl( \sum_{j=1,j \neq l}^k 1\{ y^{(i)} = j \} \ log\ a_{[j]} \biggr) \biggl] \Biggl] \\
&= -\frac{1}{m} \Biggl[ \sum_{i=1}^m \biggl[ 1\{ y^{(i)} = l \} \biggl( \frac{\partial \ (log\ a_{[l]}) }{\partial \ a_{[l]}} \cdot \frac{\partial \ a_{[l]}}{\partial \ z_{[l]}} \biggl) +  \biggl( \sum_{j=1,j \neq l}^k 1\{ y^{(i)} = j \} \cdot \frac{\partial \ (log\ a_{[j]}) }{\partial \ a_{[j]}} \cdot \frac{\partial \ a_{[j]}}{\partial \ z_{[l]}}  \biggr)  \biggl] \Biggl] \\
&= -\frac{1}{m} \Biggl[ \sum_{i=1}^m \biggl[ 1\{ y^{(i)} = l \} \biggl( \frac{1}{a_{[l]}} \cdot a_{[l]} ( 1-a_{[l]} ) \biggl) +  \biggl( \sum_{j=1,j \neq l}^k 1\{ y^{(i)} = j \} \cdot \frac{1}{a_{[j]}} \cdot (-a_{[j]} a_{[l]}) \biggr)  \biggl] \Biggl] \\
&= -\frac{1}{m} \Biggl[ \sum_{i=1}^m \biggl[ 1\{ y^{(i)} = l \} \cdot ( 1-a_{[l]} ) - \sum_{j=1,j \neq l}^k 1\{ y^{(i)} = j \} \cdot a_{[l]} \biggl] \Biggl] \\
&= -\frac{1}{m} \Biggl[ \sum_{i=1}^m \biggl[ 1\{ y^{(i)} = l \} - 1\{ y^{(i)} = l \} \cdot a_{[l]} - \sum_{j=1,j \neq l}^k 1\{ y^{(i)} = j \} \cdot a_{[l]} \biggl] \Biggl] \\
&= -\frac{1}{m} \Biggl[ \sum_{i=1}^m \biggl[ 1\{ y^{(i)} = l \} - a_{[l]} \cdot \sum_{j=1}^k 1\{ y^{(i)} = j \} \biggl] \Biggl] \\
&= -\frac{1}{m} \Biggl[ \sum_{i=1}^m \biggl[ 1\{ y^{(i)} = l \} - a_{[l]} \biggl] \Biggl] \\
\end{align}
$$

在很多的文献中，会把上式记成如下形式，其中上标(i)表示是第i个样本：
$$
\begin{align}
\delta_{[l]}^{[2](i)} &= \frac{\partial \ E^{(i)} }{\partial \ z_{[l]}^{[2](i)}} = \frac{\partial}{ \partial \ z_{[l]}^{[2](i)} } \biggl[- \sum_{j=1}^k 1\{ y^{(i)} = j \} \ log\ a_{[j]}^{[2](i)}  \biggl]  = a_{[l]}^{[2](i)} - 1\{ y^{(i)} = l \} \\
\frac{\partial}{\partial \ W_{[l]t}^{[2]}} J(W, b) &= \frac{\partial  }{\partial \ W_{[l]t}^{[2]} } (\frac{1}{m} \sum_{i=1}^m E^{(i)})  
= \frac{1}{m} \sum_{i=1}^m ( \frac{\partial \ E^{(i)} }{\partial \ z_{[l]}^{[2](i)}} \cdot \frac{\partial \ z_{[l]}^{[2](i)}}{\partial \ W_{[l]t}^{[2]}} )       
=  \frac{1}{m} \sum_{i=1}^m ( \delta_{[l]}^{[2](i)} \cdot a_t^{[1](i)} )       \\
\frac{\partial}{\partial \ b_{[l]}^{[2]}} J(W, b) &= \frac{\partial  }{\partial \ b_{[l]}^{[2]} } (\frac{1}{m} \sum_{i=1}^m E^{(i)})  
= \frac{1}{m} \sum_{i=1}^m ( \frac{\partial \ E^{(i)} }{\partial \ z_{[l]}^{[2](i)}} \cdot \frac{\partial \ z_{[l]}^{[2](i)}}{\partial \ b_{[l]}^{[2]}} )       
=  \frac{1}{m} \sum_{i=1}^m  \delta_{[l]}^{[2](i)}        \\
\end{align}
$$


> 1. 这里log的底数用的是e
> 2. 这里的x指的是输出层的上一层的输出
> 3. 关于$\delta$的原文是：for each node $i$ in layer $l$, we would like to compute an “error term” $\delta^{(l)}$ that measures how much that node was “responsible” for any errors in our output ，链接在[这里](http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/)

#### 隐藏层

隐藏层的激活函数采用的是tanh函数。

先考虑只有一个隐藏层的情况，参考下面这张图片，可以发现当前层的每个节点的权重和偏置会影响下一层的各个节点

![1](/images/learn-ml-2.png)
$$
\begin{align}
& z^{[1]} = xW^{[1]}+ b^{[1]} \\
& a^{[1]} = \sigma (z^{[1]}) = tanh(z^{(i)}) \\
& z^{[2]} = a^{[1]}W^{[2]} + b^{[2]} \\
& a^{[2]} = \hat y = softmax (z^{[2]}) \\
\end{align}
$$

$$
\begin{align}
\frac{\partial}{\partial \ W^{[1]}_{[v]t}} J(W, b) &= \frac{\partial}{ \partial \ W^{[1]}_{[v]t} } \bigl( \frac{1}{m} \sum_{i=1}^m E^{(i)} \bigr) \\
&= \frac{1}{m} \sum_{i=1}^m \biggl[ \frac{\partial}{ \partial \ W^{[1]}_{[v]t} } E^{(i)} \biggr] \\
&= \frac{1}{m} \sum_{i=1}^m \biggl[ \frac{\partial \ E^{(i)}}{ \partial \ a^{[1](i)}_{[v]} } \cdot  \frac{\partial \ a^{[1](i)}_{[v]}}{\partial \ z^{[1](i)}_{[v]}} \cdot \frac{\partial \ z^{[1](i)}_{[v]}}{\partial \ W^{[1]}_{[v]t}} \biggr] \\
&= \frac{1}{m} \sum_{i=1}^m \biggl[ ( \sum_o \biggl[ \frac{\partial \ E^{(i)} }{ \partial \ z^{[2](i)}_{[o]} } \cdot \frac{\partial \ z_{[o]}^{[2](i)} }{ \partial \ a^{[1](i)}_{[v]} } \biggr] ) \cdot \frac{\partial \ a^{[1](i)}_{[v]}}{\partial \ z^{[1](i)}_{[v]}} \cdot \frac{\partial \ z^{[1](i)}_{[v]}}{\partial \ W^{[1]}_{[v]t}} \biggr] \quad \text{(multivariate chain rule)}\\
&= \frac{1}{m} \sum_{i=1}^m \biggl[ ( \sum_o \biggl[ \delta_{[o]}^{[2](i)} \cdot W_{[o]v}^{[2]} \biggr] ) \cdot (1 - (z_{[v]}^{[1](i)})^2 ) \cdot x_t^{(i)} \biggr] \\
\end{align}
$$

$$
\begin{align}
\frac{\partial}{\partial \ b^{[1]}_{[v]}} J(W, b) &= \frac{\partial}{ \partial \ b^{[1]}_{[v]} } \bigl( \frac{1}{m} \sum_{i=1}^m E^{(i)} \bigr) \\
&= \frac{1}{m} \sum_{i=1}^m \biggl[ \frac{\partial}{ \partial \ b^{[1]}_{[v]} } E^{(i)} \biggr] \\
&= \frac{1}{m} \sum_{i=1}^m \biggl[ \frac{\partial \ E^{(i)}}{ \partial \ a^{[1](i)}_{[v]} } \cdot  \frac{\partial \ a^{[1](i)}_{[v]}}{\partial \ z^{[1](i)}_{[v]}} \cdot \frac{\partial \ z^{[1](i)}_{[v]}}{\partial \ b^{[1]}_{[v]}} \biggr] \\
&= \frac{1}{m} \sum_{i=1}^m \biggl[ ( \sum_o \biggl[ \frac{\partial \ E^{(i)} }{ \partial \ z^{[2](i)}_{[o]} } \cdot \frac{\partial \ z_{[o]}^{[2](i)} }{ \partial \ a^{[1](i)}_{[v]} } \biggl] ) \cdot \frac{\partial \ a^{[1](i)}_{[v]}}{\partial \ z^{[1](i)}_{[v]}} \cdot \frac{\partial \ z^{[1](i)}_{[v]}}{\partial \ b^{[1]}_{[v]}} \biggr] \quad \text{(multivariate chain rule)}\\
&= \frac{1}{m} \sum_{i=1}^m \biggl[ ( \sum_o \biggl[ \delta_{[o]}^{[2](i)} \cdot W_{[o]v}^{[2]} \biggl] ) \cdot (1 - (z_{[v]}^{[1](i)})^2 ) \biggr] \\
\end{align}
$$

按照输出层的$\delta$的定义，可以定义任意一层的$\delta$，并在计算的时候将本层的$\delta$传递给下一层，从而计算各层的权重和偏置的导数。

> 几个变量相互之间有依赖关系，这时某个变量的偏导数不能反映变化率，要表示在该变量上的变化率，应该使用该变量的全导数。变量相互独立时，偏导数可以表示变化率。

参考链接：
- [Total derivative]( https://en.wikipedia.org/wiki/Total_derivative#Differentiation_with_indirect_dependencies)
- [如何理解神经网络里面的反向传播算法](https://www.zhihu.com/question/24827633/answer/91489990)
- [A Step by Step Backpropagation Example](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)

#### 终止条件

- 权重的更新低于某个阈值；
- 预测的错误率低于某个阈值；
- 达到预设一定的循环次数；

#### 向量化

下面这张图描述了正向传播时，各变量的维度

![1](/images/learn-ml-3.png)

其中$x$的横线表示$x$可以看做是由一组横向量组成的，$W$的竖线表示$W$可以看做是由一组竖向量组成的。

## 附录

### 反向传播的一种错误求导

在隐藏层的反向传播求导时，下面的求导方式是错误的：
$$
\begin{align}
\frac{\partial \ E^{(i)}}{ \partial \ a^{[1](i)}_{[v]} }
&=  \sum_o \bigl[ \frac{\partial \ E_{[o]}^{(i)} }{ \partial \ z^{[2](i)}_{[o]} } \cdot \frac{\partial \ z_{[o]}^{[2](i)} }{ \partial \ a^{[1](i)}_{[v]} } \bigr]  \\
\end{align}
$$

原因是$z_{[o]}^{(i)[2]}$对所有的$E_{[j]}^{(i)}$都有影响，而这里只考虑了与$z_{[o]}^{(i)[2]}$相对应的$E_{[o]}^{(i)}$。

输出层的求导是没问题的，因为考虑了所有的分量。

### 数学复习

1. 矩阵A(m, n)，m指行数，n指列数
2. sigmoid函数求导

$$
\begin{align}
\sigma^\prime(z) &= \bigl( \frac{1}{1 + e^{-z}} \bigr)^\prime = (-1)(1+e^{-z})^{(-1)-1} \cdot (e^{-z})^\prime = \frac{1}{(1+e^{-z})^2} \cdot  (e^{-z}) \\
&= \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} = \frac{1}{1+e^{-z}} \cdot (1-\frac{1}{1+e^{-z}}) \\
&= \sigma(z)(1-\sigma(z))
\end{align}
$$

### 使用数学软件

在一些推导中，特别是矩阵的计算，想直观的看一下展开后的结果，如果完全手算，会比较费时，可以使用mathematica进行符号计算。例如，计算矩阵的相乘，在mathematica中输入以下代码

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

## 代码

下面是3层bp神经网络的python实现，取自[这里](http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/)，我做了一些修改

```python
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
```

