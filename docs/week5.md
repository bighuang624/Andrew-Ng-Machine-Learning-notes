# 第五周

## 应用机器学习的建议

### 决定下一步做什么

模型的预测错误可以通过以下方式进行排解：

* 更多训练数据：解决高方差
* 用更少的特征：解决高方差
* 用额外的特征：解决高偏差
* 用多项式特征：解决高偏差
* 增大或减小正则化参数 $\lambda$：解决高方差或高偏差

可以通过画出**学习曲线（Learning curves）**来帮助分析。

### 模型选择和训练、验证、测试集

假设函数可能在训练集上拟合得很好，但是由于过拟合，在实际使用中表现不佳。因此，我们将数据集划分为训练集、验证集和测试集，以评估假设函数。

* 验证集：对应在训练中不断更新的参数和模型准确率；
* 测试集：对应超参数和最终的模型准确率。

或者说，**验证集用于模型选择和调参，而测试集用于估计实际使用时的泛化能力**。这样，我们不需要在利用测试集调参后，又使用测试集来估计泛化能力而得到错误的结果。

### 诊断偏差和方差

偏差（欠拟合）：

- $J\_{train}(\theta)$ 较高；
- $J\_{train}(\theta) \approx J\_{train}(\theta)$。

方差（过拟合）：

- $J\_{train}(\theta)$ 较低；
- $J\_{train}(\theta) \geq J\_{train}(\theta)$。

### 正则化和偏差、方差

$$J(\theta) = \frac{1}{2m}\sum^m\_{i=1}(h\_{\theta}(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m}\sum^m\_{j=1}\theta\_j^2$$

首先考虑不使用正则化，然后选取一系列想要尝试的 $\lambda$ 值，最小化代价函数来得到 $\theta$，然后用验证集来评估，评估时用的代价函数去掉正则项：

$$J\_{cv}(\theta) = \frac{1}{2m\_{cv}}\sum^{m\_{cv}}\_{i=1}(h\_{\theta}(x^{(i)}\_{cv}) - y\_{cv}^{(i)})^2$$

可以画一幅如下图所示的图像来确定合适的 $\lambda$（实际一般有噪声）：

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/regularization-bias-and-variance.png)

## 机器学习系统设计

### 误差分析

可以将误分类的验证集数据提出来，看看哪一类误分类比例最高，做出相应的改进。

### 不对称性分类的误差评估

当正负样本的比例比较悬殊时，单纯使用正确率来评估会使得分类器只预测一种结果。因此，我们可以使用**准确率（Precision）**和**召回率（Recall）**作为评估度量。

$$P = \frac{TP}{TP+FP}$$

$$R = \frac{TP}{TP+FN}$$

两者都较高，说明模型在面对偏斜类时也有很好的表现。但大部分情况下，需要在两者之间做权衡。如果想要结果要较高的确信度，那么将准确率作为最重要的标准；如果不想漏掉样例（例如疾病判断），那么将召回率作为最重要的标准。

如果想要综合评价，我们可以使用 F1-Score：

$$F1 = \frac{2PR}{P+R}$$

<script type="text/x-mathjax-config">
 MathJax.Hub.Config({
   tex2jax: {inlineMath: [ ['$', '$'] ],
         displayMath: [ ['$$', '$$']]}
 });
</script>

<script src="https://cdn.bootcss.com/mathjax/2.7.4/latest.js?config=default"></script>