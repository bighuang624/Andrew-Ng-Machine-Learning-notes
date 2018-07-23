# 第三周

## Logistic Regression

### 二分类

$$h\_{\theta} = g(\theta^Tx)$$

$$z = \theta^Tx$$

$$g(z) = \frac{1}{1 + e^{-z}}$$

其中，$g(z)$ 被称为 Sigmoid 函数（或者 Logistic 函数），将任意实数映射到 (0, 1) 区间。这样，可以通过划分判定边界（decision boundary）进行分类。

### 成本函数

Logistic Regression 不能使用和线性回归相同的成本函数，因为 Sigmoid 函数是非凸的，容易陷入局部最优。

作为替代，使用以下成本函数：

$$J(\theta) = \frac{1}{m}\sum^m\_{i=1}Cost(h\_{\theta}(x^{(i)}), y^{(i)})$$

$$Cost(h\_{\theta}(x), y) = -y log(h\_{\theta}(x)) - (1-y)log(1-h\_{\theta}(x))$$

注意，$y$ 只有 0 或 1 两种取值。

向量化的表示为：

$$h = g(X\theta)$$

$$J(\theta) = \frac{1}{m} \cdot (-y^Tlog(h) - (1-y)^Tlog(1-h))$$

### 梯度下降

由梯度下降的一般形式：

$$\theta\_j := \theta\_j - \alpha \frac{\partial}{\partial \theta\_j}J(\theta)$$

可以推导出 LR 的梯度下降公式：

$$\theta\_j := \theta\_j - \frac{\alpha}{m} \sum^m\_{i=1}(h\_{\theta}(x^{(i)}) - y^{(i)})x\_j^{(i)}$$

向量化表示为：

$$\theta := \theta - \frac{\alpha}{m}X^T(g(X\theta) - \vec y)$$

推导过程如下：

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/derivative-of-sigmoid-function.png)

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Partial-derivative-of-J.png)

## 正则化

**高偏差（high bias）**或者欠拟合是指模型对已有数据的拟合较差，通常因为模型太简单或者使用过少的特征；而**高方差（high variance）**或者过拟合是指模型对已有数据的拟合过强，以至于对于新数据无法很好的预测，通常因为模型过于复杂。

两种技术可用于处理过拟合问题：

1. 减少特征数量：人工选择留下的特征，或者使用特征选择的算法；
2. **正则化（Regularization）**：留下所有的特征，但是减小参数 $\theta\_j$。

### 加入正则化的成本函数

$$min\_{\theta}\frac{1}{2m}\Big[\sum^m\_{i=1}(h\_{\theta}(x^{(i)}) - y^{(i)})^2 + \lambda \sum^n\_{j=1}\theta\_j^2\Big]$$

其中，$\lambda$ 用于控制参数过大所付出的代价。$\lambda$ 选取过大时，可能导致欠拟合。

### 加入正则化的线性回归

#### 梯度下降

$$\theta\_0 := \theta\_0 - \alpha \frac{1}{m}\sum^m\_{i=1}(h\_{\theta}(x^{(i)}) - y^{(i)})x\_0^{(i)}$$

$$\theta\_j := \theta\_j - \alpha \Big[\Big(\frac{1}{m}\sum^m\_{i=1}(h\_{\theta}(x^{(i)}) - y^{(i)})x\_j^{(i)} \Big) + \frac{\lambda}{m}\theta\_j \Big] \ \ \ \ j = 1, 2, ..., n$$

#### 正规方程

$$\theta = (X^TX + \lambda \cdot L)^{-1}X^Ty$$

$$where  \ L= 
  \begin{bmatrix}
   0 &   &   \\\
     & 1 &   \\\
     &   & \ddots  \\\
     &   &   & 1
  \end{bmatrix}$$

### 加入正则化的 Logistic Regression

#### 成本函数

$$J(\theta) = -\frac{1}{m}\sum^m\_{i=1}\Big[y ^{(i)}log(h\_{\theta}(x^{(i)})) + (1-y^{(i)})log(1-h\_{\theta}(x^{(i)}))\Big] + \frac{\lambda}{2m}\sum^n\_{j=1}\theta^2\_j$$

#### 梯度下降

$$\theta\_0 := \theta\_0 - \alpha \frac{1}{m}\sum^m\_{i=1}(h\_{\theta}(x^{(i)}) - y^{(i)})x\_0^{(i)}$$

$$\theta\_j := \theta\_j - \alpha \Big[\Big(\frac{1}{m}\sum^m\_{i=1}(h\_{\theta}(x^{(i)}) - y^{(i)})x\_j^{(i)} \Big) + \frac{\lambda}{m}\theta\_j \Big] \ \ \ \ j = 1, 2, ..., n$$

<script type="text/x-mathjax-config">
 MathJax.Hub.Config({
   tex2jax: {inlineMath: [ ['$', '$'] ],
         displayMath: [ ['$$', '$$']]}
 });
</script>

<script src="https://cdn.bootcss.com/mathjax/2.7.4/latest.js?config=default"></script>