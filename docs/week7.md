# 第七周

## 降维



## 异常检测

**异常检测（Anomaly Detection）**是机器学习算法的一个常见应用。这是一个无监督学习问题，会通过新的数据特征落在已有数据的特征分布的概率来判断新的数据是否隐含了某种异常。

### 高斯分布

高斯分布（Gaussian ditribution），即正态分布（Normal ditribution）。

假设 $x \in \mathbb{R}$，如果 $x$ 的概率分布服从高斯分布，其中均值为 $\mu$，方差为 $\sigma^2$，则写作

$$x \sim \mathcal{N}(\mu, \sigma^2)$$

概率分布的公式为：

$$p(x;\mu, \sigma^2) = \frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(x-\mu)^2}{2\sigma^2})$$

参数估计（parameter estimation）问题是，给定数据集确定满足高斯分布，需要估计出 $\mu$ 和 $\sigma$。计算公式为：

$$\mu = \frac{1}{m}\sum^m\_{i=1}x^{(i)}$$

$$\sigma^2 = \frac{1}{m}\sum^m\_{i=1}(x^{(i)} - \mu)^2$$

其中，$\frac{1}{m}$ 有时也写作 $\frac{1}{m-1}$，在实践中区别不大。

### 算法

假设有一个训练集 $\big\\{ x^{(1)},...,x^{(m)} \big\\}$，每个样本 $x \in \mathbb{R}^n$。

在特征相互独立的假设下，有

$$p(x) = \prod^n\_{j=1}p(x\_j;\mu\_j,\sigma\_j^2)$$

如果 $p(x) < \varepsilon$，认为异常。$\varepsilon$ 需要选定。

分布项 $p(x)$ 的估计问题有时称为**密度估计（Density estimation）**问题。

### 开发和评估异常检测系统

如果有一些带标签的数据，来指明哪些是正常数据，哪些是异常数据，则这些数据可以用于评估算法。

我们这样划分训练集、验证集和测试集：

* 训练集：正常无标签数据
* 验证集和测试集：包含少量已知是异常的数据

系统开发步骤：

1. 使用训练集拟合模型 $p(x)$；
2. 对于验证集和测试集用模型进行预测；
3. 因为数据类别比例相差较大，可以用 F1-Score 等来评估算法。

选择 $\varepsilon$ 的一种方法是使用多个值，最后选择能够最大化 F1-Score 的 $\varepsilon$ 值。

### 异常检测 VS 监督学习

异常检测与监督学习的区别：

1. 异常检测中样本比例不平衡，而监督学习一般有比例相近的正负样本；
2. 异常检测不追求从样本中分析出异常的具体种类，因为实际中可能出现与以前完全不同的全新异常。而监督学习会划分具体种类。

### 选择使用的特征

1. 通过画直方图选择比较符合高斯分布的特征。不太符合的可以通过一些转换（例如取对数）使其符合高斯分布。
2. 选择异常样本中值相对于正常样本概率较低的特征。这样，可以使异常样本得到相对较低的 $p(x)$。反映到实际中选择，可以是异常样本中某特征值异常地大或异常地校。

### 多元高斯分布

**多元高斯分布（Multivariate Gaussian distribution）**有两个参数 $\mu$ 和 $\Sigma$，$\mu \in \mathbb{R}$，$\Sigma \in \mathbb{R}^2$。

$$p(x;\mu, \Sigma) = \frac{1}{(2\pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}}exp \Big( -\frac{1}{2}(x-\mu)^T \Sigma^{-1}(x-\mu)\Big)$$

参数估计：

$$\mu = \frac{1}{m}\sum^m\_{i=1}x^{(i)}$$

$$\Sigma = \frac{1}{m}\sum^m\_{i=1}(x^{(i)}-\mu)(x^{(i)}-\mu)^T$$

使用原先的高斯分布的模型需要创建新的特征来捕捉异常的特征组合值。而多元高斯分布能够自动捕捉不同特征之间的关系；但原始模型的计算成本较低，可以使用数量巨大的特征。而多元高斯分布需要计算协方差矩阵 $Sigma$ 的逆矩阵，因此计算成本较高的同时，需要保证样本数量大于特征数量，来保证 $Sigma$ 可逆。

<script type="text/x-mathjax-config">
 MathJax.Hub.Config({
   tex2jax: {inlineMath: [ ['$', '$'] ],
         displayMath: [ ['$$', '$$']]}
 });
</script>

<script src="https://cdn.bootcss.com/mathjax/2.7.4/latest.js?config=default"></script>