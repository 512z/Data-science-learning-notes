# Machine learning - WHAT

## 1. data
1. inference our data
i.e. understand properties associated to our data
2. understand distribution of data
e.g. to generate/simulate more data. Or, to simply understand its properties
## 2. models
1. Which models to use to solve a task?
2. How to engineer/select a suitable model?
3. Which models are out there?
## 3. optimisation targets
1. which question shall our model solve (with our data)
2. which criteria shall we require from our model
## 4. optimisation algorithm
1. how do we adapt/update our model parameters using our data to solve the targets?

-----
# 2024 Nobel Prize

- Hopfield创建了一种可以存储和重构信息的结构（即Hopfield网络）
- Hinton发明了一种可以独立发现数据中性质的方法，这个方法对现今使用的大型神经网络非常重要

1. 物理学理论基础：

- Hopfield网络的设计灵感来自于物理学中的自旋玻璃系统（spin glass）理论
- 他使用了物理学中的能量状态概念，将神经网络类比为一个物理系统，网络的学习过程可以被描述为能量最小化的过程
- 这种方法借鉴了统计物理学中的玻尔兹曼分布理论

2. 跨学科应用：

- 他们的工作展示了物理学原理如何可以用来解决信息处理和计算的问题
- 这代表了物理学在跨领域应用中的重要性，尤其是在复杂系统的建模和理解方面

3. 物理学思维方式：

- 他们采用了物理学的研究方法：建立模型、寻找基本规律、验证假设
- 将大脑的信息处理机制抽象为可以用物理方程描述的系统

------

# Machine learning - Challenges

## 1. Number of data points - big data regime
## 2. High-dimensional data - complexity
## 3. Number of model parameters - model capacity

------

# Introduction
## 数据和模型
![[Pasted image 20241029225154.png]]

在第一张图中，我们看到：

1. 数据模型： $$y_i = f_{true}(x_i) + \eta_i$$ 其中 $$\eta_i \sim N(0,\sigma)$$ (高斯噪声)


2. 模型函数示例： $$f_1(x,\theta) = \theta_1 x + \theta_0$$ $$f_2(x,\theta) = \theta_0 + \theta_1 x + \theta_2 x^2$$
这段内容在讨论函数近似和模型选择的问题。让我来详细解释一下:

1. $$\mathcal{F}$$ 表示函数族(function family)或假设空间(hypothesis space),即我们可以选择的所有可能的函数的集合。

2. $$f_{true}$$ 表示真实的目标函数,即我们希望近似的理想函数。

3. 这段话说"我们经常在不知道 $$f_{true}$$ 的情况下选择函数族 $$\mathcal{F}$$"。这是机器学习中的一个常见情况 - 我们并不知道真实的数据生成过程或目标函数的具体形式。

4. "因此可能发生: $$f_{true} \notin \mathcal{F}$$" 意思是真实函数可能不在我们选择的函数族中。这就是所谓的模型错配(model misspecification)问题。

这揭示了机器学习中的一个基本挑战:
- 我们需要选择一个函数族来近似未知的目标函数
- 但我们选择的函数族可能不包含真实函数
- 这会导致不可避免的近似误差(approximation error)

这种情况在实践中很常见,例如:
- 用线性模型去拟合非线性数据
- 用有限维参数的神经网络去近似可能是无限维的真实函数
- 用离散的决策树去近似连续的决策边界

这就是为什么在机器学习中,我们需要:
1. 仔细选择模型族
2. 评估模型的泛化能力 
3. 权衡模型复杂度和拟合能力

![[Pasted image 20241029225205.png]]
在第二张图中，我们看到：

1. 无限参数函数： $$f_\infty(x) = \sum_{i=0}^{\infty} \theta_i x^i$$

这些方程代表：

1. 真实的数据生成过程 - 观测值(y_i)由真实函数加上随机噪声生成
2. 简单的参数化模型 - 由θ参数化的线性和二次函数
3. 一个具有无限参数的复杂模型，理论上可以近似任何连续函数

这里的关键见解是：虽然我们可能不知道真实函数f_true，但通过使用足够表达能力的模型类（如神经网络或无限参数展开），我们可以避免f_true不在我们的函数类F中的问题。但是，这会导致过拟合和模型复杂性问题，需要仔细管理。

Neural Networks(神经网络)的优势在于它们可以作为"通用近似器"，能够学习和逼近几乎任何复杂的函数关系。不过，为了避免过拟合，我们通常需要：

1. 适当的正则化技术
2. 合适的模型架构设计
3. 充足的训练数据
4. 良好的验证策略

这就是为什么在实践中，我们需要在模型复杂性和泛化能力之间找到一个良好的平衡点。

----
## 如何得到参数?

1. 目标是找到最适合数据的函数。这通过最小化损失函数(loss function)来实现。
2. 这里展示了一个常见的损失函数 - 均方误差(Mean Squared Error, MSE):

$$E(θ) = \frac{1}{2N} \sum_{i=1}^N (f_n(x_i, θ) - y_i)^2$$

其中:

- $N$ 是样本数量
- $x_i$ 是输入数据
- $y_i$ 是真实标签
- $f_n(x_i, θ)$ 是模型预测值
- $θ$ 是模型参数

3. 对于线性回归问题,预测函数形式为:

$$f_n(x, θ) = θ_0 + x θ_1$$

其中:

- $θ_0$ 是偏置项(bias)
- $θ_1$ 是权重(weight)

4. 将这个预测函数代入损失函数,得到完整形式:

$$E(θ) = \frac{1}{2 \cdot 2}[(θ_0 + x_1θ_1 - y_1)^2 + (θ_0 + x_2θ_1 - y_2)^2]$$

这个损失函数量化了模型预测值与真实值之间的差距。通过最小化这个损失函数(使用梯度下降等优化方法),我们可以找到最优的参数 $θ_0$ 和 $θ_1$。

这就是机器学习中参数学习的基本原理。实际应用中可能会使用更复杂的模型结构和损失函数,但基本思想是相同的。

-----
## 优化问题

1. 优化问题的定义:
优化问题是:
$$\hat{\theta} = \arg\min_{\theta} E(\theta)$$
这是机器学习中最基本的优化形式,其中:
- $\theta$ 代表模型参数
- $E(\theta)$ 代表误差/损失函数
- $\hat{\theta}$ 代表最优参数

2. 主要问题:
工作中我们经常面临:
- 有限的数据点(finite amount of data points)
- 需要选择具有许多参数的好模型(good models with many parameters)

3. 解决方案包含三个步骤:
a) 将数据集分为训练集和测试集(Split dataset into training and test set)
- 这是为了评估模型的泛化能力

b) 在训练数据上解决优化问题(Solve optimization problem on training data)
- 使用各种优化算法找到最优参数

c) 在测试数据上评估/比较模型(Compare/evaluate models on test data)
- 验证模型是否过拟合
- 确保模型在未见过的数据上也能表现良好

4. 图示说明:
底部的曲线图展示了模型复杂度与性能之间的权衡:
- x轴代表模型复杂度
- y轴代表误差
- 需要找到平衡点,使模型既不欠拟合也不过拟合

这个例子很好地说明了机器学习中的一个核心原则:我们不仅要让模型在训练数据上表现好,更要确保它能在测试数据上泛化良好。

---
# bias v.s. variance

1. 基本设定：

- 数据生成过程：$y = f(x) = p(x) + \eta_{noise}$
- 其中 $f(x)$ 是真实函数，$p(x)$ 是解析部分，$\eta_{noise}$ 是噪声
- 假设：数据来自足够复杂的函数，我们无法完全学习 $f(x)$

## 随着数据量增加, train和test的error变化

![[Pasted image 20241029221055.png]]

第一张图 - 固定模型复杂度，改变数据量：

- x轴：数据点数量
- y轴：误差
- 两条关键曲线：
    - $E_{train}$：训练误差
    - $E_{test}$：测试误差
- 观察：
    - 随着数据量增加，训练误差逐渐上升
    - 测试误差逐渐下降
    - 两条曲线最终趋于收敛
- 两个关键概念：
    - Variance（方差）：模型对不同训练数据集的敏感度
    - Bias（偏差）：模型预测值与真实值的系统性偏离

1. Bias（偏差）变化：

- 随着数据点增加，bias基本保持稳定或略有下降
- 原因：bias反映的是模型的拟合能力，主要由模型的复杂度决定
- 该图中，通过训练误差($E_{train}$)曲线的渐近值可以看出bias的大小
- 即使有无限多的数据点，只要模型复杂度固定，bias就会存在一个下限

2. Variance（方差）变化：

- 随着数据点增加，variance显著降低
- 这可以从测试误差($E_{test}$)和训练误差($E_{train}$)之间的差距逐渐减小看出
- 原因：
    - 更多的数据点能更好地约束模型参数
    - 减少了模型对单个数据点的过度依赖
    - 提高了模型的稳定性和泛化能力

3. 总体效果：

- 当数据点很少时：
    - variance大：模型对训练数据的微小变化非常敏感
    - 训练误差低但测试误差高
- 当数据点增加时：
    - variance降低：模型变得更稳定
    - 训练误差和测试误差逐渐收敛
    - 最终误差主要由bias决定

4. 数学表达： 对于固定复杂度的模型，总体误差可以分解为： $$E_{total} = Bias^2 + Variance + \text{Irreducible Error}$$ 其中：

- $Bias^2$ 随数据量变化不大
- $Variance$ 随数据量增加而减小
- Irreducible Error 是由数据本身的噪声($\eta_{noise}$)决定的，不随数据量变化

这就解释了为什么增加数据量能提高模型性能，但是无法完全消除误差 - 因为bias和不可约误差的存在。
### code
```python

# For fixed model complexity, plot the train and test error as the number of data points varies.

  
# true function

def true_function(x):

return np.sin(x) + 0.1 * x**2

  
# generate data points with noise

def generate_data(n_samples):

x = np.linspace(-5, 5, n_samples) # generate n_samples points between -5 and 5

y = true_function(x) + np.random.normal(0, 0.1, n_samples) # generate n_samples y values, based on the true function with some noise

return x, y

  
# polynomial model

def fit_polynomial(x, y, degree):

poly_features = PolynomialFeatures(degree=degree, include_bias=False)

x_poly = poly_features.fit_transform(x.reshape(-1, 1)) # transform x to polynomial features

model = LinearRegression()

model.fit(x_poly, y)

return model, poly_features

  
# Mean Squared Error

def calculate_error(model, poly_features, x, y):

x_poly = poly_features.transform(x.reshape(-1, 1))

y_pred = model.predict(x_poly)

return np.mean((y - y_pred)**2)

  
# calculate bias and variance

def calculate_bias_variance(models, poly_features, x, y_true):

x_poly = poly_features.transform(x.reshape(-1, 1))

predictions = np.array([model.predict(x_poly) for model in models])

avg_prediction = np.mean(predictions, axis=0)

# Bias^2 = (E[y_pred] - f(x))^2

bias = np.mean((avg_prediction - y_true) ** 2)

# Variance = E[(y_pred - E[y_pred])^2]

variance = np.mean(np.var(predictions, axis=0))

return bias, variance

  
# set the range of number of data points

n_samples_range = np.arange(10, 2000, 100)

fixed_degree = 3 # fixed model complexity

  
train_errors = []

test_errors = []

  
# iterate over the number of data points

for n_samples in n_samples_range:

# generate data

x, y = generate_data(n_samples)

# split data into training and testing sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# fit polynomial model

model, poly_features = fit_polynomial(x_train, y_train, fixed_degree)

# calculate train and test errors

train_errors.append(calculate_error(model, poly_features, x_train, y_train))

test_errors.append(calculate_error(model, poly_features, x_test, y_test))

  
# plot the errors

plt.figure(figsize=(10, 6))

plt.plot(n_samples_range, train_errors, label='Train Error', marker='o')

plt.plot(n_samples_range, test_errors, label='Test Error', marker='o')

plt.xlabel('Number of Data Points')

plt.ylabel('Mean Squared Error')

plt.title('Error vs Number of Data Points (Fixed Model Complexity)')

plt.legend()

plt.grid(True)

plt.show()


```

![[Pasted image 20241029221018.png]]

----

## 随着模型参数增加, train和test的error变化

![[Pasted image 20241029221110.png]]

1. 第二张图 - 固定数据集，改变模型复杂度：

- x轴：模型参数数量（复杂度）
- y轴：误差
- 两条关键曲线：
    - Variance曲线：随复杂度增加而增加
    - Bias曲线：随复杂度增加而减少
- 关键区域：
    - 区域I（模型过于简单）：高偏差，低方差
    - 区域II（模型过于复杂）：低偏差，高方差
    - 最优复杂度：在两条曲线交叉处附近

2. 这些曲线对机器学习至关重要，因为它们：

- 帮助我们理解模型选择的trade-off
- 指导我们选择合适的模型复杂度
- 解释过拟合和欠拟合现象
- 提供了评估和改进模型的框架

### code

```python
# Set the fixed number of data points

n_samples = 1000


# Initialize lists to store bias and variance

biases = []

variances = []

  

# Range of polynomial degrees to test

degrees = range(1, 10)

  

# Iterate over polynomial degrees

for degree in degrees:

	# List to store predictions from different models
	
	degree_biases = []
	
	degree_predictions = []
	
	# Perform multiple training runs with different data to capture variance
	
	for _ in range(100): # Use 100 different samples for variance estimation
	
		x, y = generate_data(n_samples)
		
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None)
		
		# No fixed random state
		
		model, poly_features = fit_polynomial(x_train, y_train, degree)
		
		# Predict on the test set
		
		y_test_predict = model.predict(poly_features.transform(x_test.reshape(-1, 1)))
		
		bias_squared = np.mean((y_test - y_test_predict) ** 2)
		
		degree_biases.append(bias_squared)
		
		degree_predictions.append(y_test_predict)
	
	avg_bias = np.mean(degree_biases)
	
	biases.append(avg_bias)
	
	variances.append(np.mean(np.var(degree_predictions, axis=0)))

  

# Plot bias and variance

plt.figure(figsize=(10, 6))

plt.plot(degrees, biases, label='Bias^2', marker='o')

plt.plot(degrees, variances, label='Variance', marker='o')

plt.xlabel('Model Complexity (Polynomial Degree)')

plt.ylabel('Error')

plt.title('Bias-Variance Trade-off')

plt.legend()

plt.grid(True)

plt.show()


```

![[Pasted image 20241029221307.png]]

---
## bias-variance量化分析

偏差-方差分解的推导过程：

1. 初始设定：
- estimator：$f(x,\hat{\theta})$ 具有从有限训练数据集得到的最优参数
- 数据生成过程随机：$(x,y) \sim p(x,y)$
- 标签包含内在噪声：$y = f(x) + \varepsilon$

2. 期望损失表达：
$$\mathbb{E}_{D,\varepsilon}(E) = \iint E(y,f(x,\hat{\theta}))p(x,y)dxdy$$

3. 使用均方误差(MSE)：
$$MSE = \iint(f(x,\hat{\theta})-y)^2p(x,y)dxdy$$

---

均方误差: $$MSE = \frac{1}{n}\sum_{i=1}^n(y_i - \hat{y}_i)^2$$ 其中：

- $y_i$ 是真实值
- $\hat{y}_i$ 是预测值
- $n$ 是样本数量
----


4. 假设：
- $p(x,y) = p(x)p(\varepsilon)$ (x和噪声独立)
- $y = f(x) + \varepsilon$

5. 中间步骤展开：
$$\int(f(x,\hat{\theta})-f(x)+f(x)-y)^2p(x)p(\varepsilon)dxd\varepsilon$$

6. 平方展开得到三项：
$$ \int[(f(x,\hat{\theta})-f(x))^2 + 2(f(x,\hat{\theta})-f(x))(f(x)-y) + (f(x)-y)^2]p(x)p(\varepsilon)dxd\varepsilon$$

7. 中间项积分：
- $\int 2(f(x,\hat{\theta})-f(x))\varepsilon p(x)p(\varepsilon)dxd\varepsilon = 0$
因为 $\int \varepsilon p(\varepsilon)d\varepsilon = 0$

> [!NOTE]
> 1. 首先分析这个积分的结构： $$\int 2(f(x,\hat{\theta})-f(x))\varepsilon p(x)p(\varepsilon)dxd\varepsilon$$
> 2. 可以通过积分的性质拆分：
>- 因为这是二重积分，我们可以先对$\varepsilon$积分，再对$x$积分
>
>$f(x,\hat{\theta})-f(x)$和$p(x)$都不依赖于$\varepsilon$，可以看作常数
> 3. 拆分后的形式： $$\int \bigg[2(f(x,\hat{\theta})-f(x))p(x)\bigg] \bigg[\int \varepsilon p(\varepsilon)d\varepsilon\bigg] dx$$
> 4. 关键点：$\int \varepsilon p(\varepsilon)d\varepsilon = 0$
>- 这是因为$\varepsilon$是噪声项
>- $p(\varepsilon)$是噪声的概率分布
>- 通常假设噪声是零均值的，即噪声的期望为0
>- 常见的例子：高斯噪声$\mathcal{N}(0,\sigma^2)$
> 5. 所以：
>- 内部积分$\int \varepsilon p(\varepsilon)d\varepsilon = 0$
>- 任何数乘以0还是0
>- 因此整个二重积分结果为0
> 

8. 考虑第三项：

- $(f(x)-y)^2 = (f(x)-(f(x)+\varepsilon))^2 = \varepsilon^2$
- $\int\varepsilon^2p(\varepsilon)d\varepsilon = \sigma_\varepsilon^2$（噪声方差）

9. 考虑第一项：

- 用期望和方差的关系：$E[X^2] = Var(X) + (E[X])^2$
- 令 $X = f(x,\hat{\theta})-f(x)$
- 则 $(f(x,\hat{\theta})-f(x))^2$ 的期望可以写成：
    - $Var(f(x,\hat{\theta})-f(x)) + (E[f(x,\hat{\theta})-f(x)])^2$

10. 综合上述三项：

- 第一项：$Var(f(x,\hat{\theta})-f(x)) + (E[f(x,\hat{\theta})-f(x)])^2$
- 第二项：0
- 第三项：$\sigma_\varepsilon^2$

11. 最终分解：
$$\mathbb{E}(E) = \sigma_\varepsilon^2 + \underbrace{Var(f(x,\hat{\theta})-f(x))}_{\text{Variance}} + \underbrace{(\mathbb{E}[f(x,\hat{\theta})-f(x)])^2}_{\text{Bias}^2}$$

其中：
- $\sigma_\varepsilon^2$：不可约误差（噪声方差）
- Variance项：estimator由于有限样本量导致的波动
- Bias项：estimator在无限数据限制下与真实函数的期望偏差

这个分解显示了模型误差的三个来源：
1. 数据本身的噪声（不可避免）
2. 模型对训练数据的敏感度（方差）
3. 模型的系统性偏差（偏差）

---
## bias-variance方差分解可视化

![[Pasted image 20241029224308.png]]

这张图展示了偏差-方差分解的可视化：

1. 图的基本结构：
- x轴：数据集大小(data size)
- y轴：损失(loss)
- $E_{test}$：测试误差曲线
- 三个关键组成部分从下到上：$\sigma_\varepsilon^2$, $\text{Bias}^2$, $\text{Var}(A)$

2. 各个组成部分：

a) 不可约误差($\sigma_\varepsilon^2$)：
- 数据本身固有的噪声
- 不随数据量变化
- 是误差的下界

b) 偏差项($\text{Bias}^2 = (E[A])^2$)：
- 反映模型的系统性误差
- 主要由模型复杂度决定
- 相对稳定，不太随数据量变化

c) 方差项($\text{Var}(A)$)：
- 随数据量增加而减小
- 反映模型对训练数据的敏感度
- 数据量越大，估计越稳定

3. 关键观察：

a) 测试误差($E_{test}$)的特点：
- 随数据量增加而减小
- 渐近趋近于$\text{Bias}^2 + \sigma_\varepsilon^2$
- 下降速度主要受方差项控制

b) 最优模型的选择：
- 需要在偏差和方差之间取得平衡
- 过于简单的模型：高偏差，低方差
- 过于复杂的模型：低偏差，高方差

4. 实践意义：
- 这个分解可以empirically（经验地）评估不同模型
- 可以通过多个样本评估方差
- 模型的最终表现($\hat{\theta}$)依赖于：
  - 模型类别的选择
  - 参数的优化
  - 数据量的大小

这个分解框架帮助我们：
1. 理解模型错误的来源
2. 指导模型复杂度的选择
3. 判断是否需要更多数据
4. 评估模型改进的方向

---
# Optimisers - 优化器

## 寻找cost(loss) function 的minimum

![[Pasted image 20241029224912.png]]

1. 优化问题的基本难点：

A. 高维参数空间：

- 现代深度学习模型通常有$O(10^6)$量级的参数
- 参数空间维度高使得搜索空间巨大
- 容易出现维度灾难(curse of dimensionality)

B. 计算成本：

- 损失函数计算代价高
- 梯度计算更加耗时
- 二阶导数（Hessian矩阵）计算在高维情况下几乎不可行

C. 非凸优化问题：

- 图中展示了凸函数(convex)和非凸函数(non-convex)的区别
- 凸函数：只有一个全局最小值，相对容易优化
- 非凸函数：存在多个局部最小值，优化更加困难
- 深度学习中的损失函数通常是非凸的

2. 常用解决方案：

梯度下降法(Gradient Descent)及其变体：

- 基于一阶导数信息
- 计算相对高效
- 可以通过各种改进来处理局部最小值问题
    - 随机梯度下降(SGD)
    - 动量法(Momentum)
    - AdaGrad/Adam等自适应方法

3. 实践考虑：

- 需要在优化效果和计算效率之间做权衡
- 选择合适的初始化策略
- 使用适当的学习率调度策略
- 考虑使用正则化技术来改善优化景观

---
## Gradient Descent - 梯度下降

梯度下降优化算法(Gradient Descent)。

梯度下降的核心思想：
- 通过沿着损失函数的负梯度方向迭代更新参数，来寻找损失函数的局部最小值
- 右图中的损失函数表示为:
$$ E(\theta) = \sum_{i=1}^n e_i(x_i,\theta) $$
其中 $e_i$ 是样本 $i$ 的误差项,可以表示为 $(g(x_i,\theta)-y_i)^2$

更新公式：
$$ \theta_{t+1} = \theta_t - \eta \nabla_\theta E(\theta_t) $$

其中:
- $\theta_t$ 是第 $t$ 次迭代的参数
- $\eta$ 是学习率(learning rate),控制每次更新的步长
- $\nabla_\theta E(\theta_t)$ 是损失函数相对于参数的梯度

梯度下降的特点：
1. 选择合适的学习率很重要:
- 太大会导致震荡或发散
- 太小会导致收敛太慢

2. 可能陷入局部最小值:
- 图中展示了参数空间中的损失函数曲线
- 红色箭头表示参数更新的方向
- 最终可能停在局部最小值处而非全局最小

3. 训练成功与否很大程度依赖于:
- 优化器的选择
- 学习率的设置
- 初始参数的选择

这是深度学习中最基础和最常用的优化算法之一,后续改进的优化器如Adam、RMSprop等都是在此基础上发展而来。


---


# Loss functions

# ML

## What do ML algorithms do?

  

- Build a model from our "training" (or learning) sample

- Based on a set of features in the data:

  

$$\vec{x}_i = \{ x_1, x_2, \dots, x_N \}$$

  

- the shape of $\vec{x}$ will be `(nevents, nfeatures)` often called `X`

  

- Learn the form of function $f$ to produce output $y$

  

$$ y_i = f(\vec{x}_i) $$

  

- $y$ can also be a vector (called a multi-class or multi-output algorithm)

  

- The algorithm "learns" by comparing its predicted output $y$ to some known output (at least in **supervised** learning)

# Features

理解features（特征）是掌握机器学习的关键。让我详细解释并举例说明：

特征是描述数据样本的独立变量或属性。它们是用来预测或分类的输入信息。

举例说明：

1. 房价预测：
   特征可能包括：
   - 房屋面积（平方米）
   - 卧室数量
   - 浴室数量
   - 房龄（年）
   - 地理位置（邮政编码）
   - 学区质量（评分）

2. 垃圾邮件分类：
   特征可能包括：
   - 邮件中特定词语的出现频率
   - 发件人地址
   - 包含链接的数量
   - 邮件大小
   - 是否包含附件

3. 医疗诊断（如糖尿病预测）：
   特征可能包括：
   - 年龄
   - 体重指数（BMI）
   - 血压
   - 血糖水平
   - 家族病史

4. 图像识别（如猫狗分类）：
   特征可能包括：
   - 像素值
   - 边缘检测结果
   - 纹理信息
   - 颜色直方图

5. 客户流失预测：
   特征可能包括：
   - 客户使用产品的频率
   - 客户服务互动次数
   - 账户年限
   - 最近购买时间
   - 消费金额

在机器学习中，算法利用这些特征来学习模式并做出预测。特征的选择和工程（即创建新的、更有信息量的特征）对模型性能至关重要。好的特征应该与预测目标相关，并能提供有区分度的信息。

理解x的shape为(n_events, n_features)是很重要的，这反映了机器学习中数据的标准组织方式。让我详细解释一下：

1. n_events（事件数量）:
   - 这代表数据集中的样本数量。
   - 每个"事件"是一个独立的观察或数据点。
   - 在机器学习中，这通常对应于数据集的行数。

2. n_features（特征数量）:
   - 这代表每个样本的属性或特征数量。
   - 每个特征是描述样本的一个独立变量。
   - 在机器学习中，这通常对应于数据集的列数（不包括目标变量）。

3. 为什么是这种形状：
   - 这种结构使得每一行代表一个完整的数据样本。
   - 每一列代表一个特定的特征，跨所有样本。

4. 实例说明：
   假设我们有一个房价预测的数据集：
   - n_events = 1000（1000个房屋样本）
   - n_features = 5（面积、卧室数、浴室数、房龄、学区评分）
   
   数据结构会是这样：
   ```
   x = [
       [120, 3, 2, 10, 8],  # 房屋1
       [150, 4, 3, 5, 9],   # 房屋2
       [100, 2, 1, 15, 7],  # 房屋3
       ...                  # 更多房屋
       [180, 5, 4, 2, 10]   # 房屋1000
   ]
   ```
   这个x的shape就是(1000, 5)。

5. 优势：
   - 便于向量化操作，提高计算效率。
   - 符合大多数机器学习库的输入格式要求。
   - 便于特征的添加、删除和操作。

6. 在Python中：
   这种结构通常用NumPy数组或Pandas DataFrame表示。例如：
   ```python
   import numpy as np
   x = np.array([[120, 3, 2, 10, 8],
                 [150, 4, 3, 5, 9],
                 [100, 2, 1, 15, 7]])
   print(x.shape)  # 输出: (3, 5)
   ```

理解这种数据结构是处理机器学习问题的基础，它影响到数据预处理、模型训练和结果解释的各个方面。


# overfitting

**过拟合 (Overfitting)**

  

过拟合是机器学习模型在训练过程中常见的问题之一，指的是模型在训练数据上表现良好，但在测试数据或新数据上表现不佳。这是因为模型“记住”了训练数据中的细节和噪声，而不是学到了数据的整体趋势。

  

**过拟合的特征**

  

• **训练误差（Training Error）很低，但测试误差（Test Error）较高**：这表明模型已经非常好地拟合了训练数据，但泛化能力（Generalization Ability）较差。

• **模型复杂度过高**：当模型的复杂度（如参数数量、非线性项、深度等）较高时，模型可能会过度拟合训练数据中的细节，导致对训练数据的“过度学习”。

• **模型对数据中的噪声过于敏感**：过拟合的模型会将训练数据中的噪声也当作特征来学习，从而影响其对新数据的预测能力。

  

**解决过拟合的方法**

  

1. **降低模型复杂度**

通过减少模型的参数数量、移除高次项或者减少网络层数来降低模型复杂度。例如，在图中左侧，蓝色的“Underfit”曲线是一个低复杂度模型，只能拟合到数据的整体趋势，而忽略了细节。

2. **增加训练数据量**

当训练数据不足时，模型容易因为数据量少而对每个样本的细节进行过度学习。增加训练数据可以帮助模型学到更为泛化的特征，减少过拟合的风险。

3. **使用正则化方法（Regularization）**

正则化方法（如L1、L2正则化）在损失函数中加入惩罚项，限制模型参数的大小，从而降低模型复杂度，减少过拟合。例如，在神经网络中加入 Dropout 层也是一种常见的正则化方法。

4. **交叉验证（Cross-validation）**

通过交叉验证技术（如K折交叉验证）来检测模型的性能，选择最优的超参数，防止模型在某一组数据上表现过于良好而导致过拟合。

5. **数据增强（Data Augmentation）**

在图像和自然语言处理中，可以对训练数据进行一定程度的变换和扩展，如图像旋转、缩放、裁剪，或句子中的词语替换等，来增强模型的泛化能力。

  

**图中的解释**

  

图中左图显示了三种拟合状态：

  

• **Underfit（欠拟合）**：蓝色直线表示一个低复杂度的模型（如线性模型），它无法捕捉数据的趋势。欠拟合意味着模型对训练数据和测试数据的预测能力都很差。

• **Overfit（过拟合）**：橙色曲线表示一个高复杂度模型（如多项式高次项模型），它非常好地拟合了训练数据中的每一个点，但在测试数据上表现很差。

• **Data（数据）**：黑点表示真实的数据分布。

  

右图显示了随着训练的进行，训练集和测试集的性能指标（如AUC）的变化。可以看到，训练集的AUC值逐渐增高并趋于稳定，而测试集的AUC值在某个点后开始下降，这就是典型的过拟合现象。

  

**数学公式表示**

  

对于一个模型而言，其拟合的目标函数可以表示为：

  

$$ y_i = f(\vec{x_i}; \vec{w}, \vec{b}) $$

  

其中：

  

• 是第  个样本的目标值。

• 是第  个样本的输入特征向量。

• 是模型的权重向量。

• 是模型的偏置项。

  

模型的损失函数（Loss Function）通常是训练误差和正则化项的加权和，如下：

  

$$ E(\vec{w}) = \sum_{i=1}^{N} \left( f(\vec{x_i}; \vec{w}, \vec{b}) - y_i \right)^2 + \lambda \sum_{j=1}^{M} w_j^2 $$

  

其中：

  

• 第一个求和项表示模型在训练数据上的均方误差（MSE）。

• 第二个求和项表示模型参数的 L2 正则化项，其中  是正则化系数，控制正则化项的权重。

  

**总结**

  

过拟合是机器学习模型训练中的常见问题，为了防止过拟合，我们可以降低模型复杂度、增加数据量、使用正则化和交叉验证等方法。理解过拟合现象以及如何解决它，是提高模型泛化能力、提升模型在测试数据上性能的重要一步。

## Problem

Given training data: 
$$
\{(x_1, y_1), ..., (x_n, y_n)\}
$$

We want to minimize:
$$
E_{\text{train}}(\hat{\theta})
$$

### Overfitting

- \( E_{\text{train}}(\hat{\theta}) \) is small but
- \( E_{\text{test}}(\hat{\theta}) \) is large.
  
This leads to the conclusion that:
- Predictive power of the model is small.

### Note

We will see many instances of models with many parameters, which give good results on the test data.

![[Pasted image 20241015124047.png]]
overfitting - poor performance outside of training regime

fixed model (complexity / parameters), varying data points

assumption: data from sufficiently complicated function i.e. we cannot exactly learn f(x)

**Understanding the Plot**

  

This plot shows several curves fitting the data points (blue circles). The x-axis represents the input feature x, and the y-axis represents the predicted values y. The three different curves represent models of varying complexity:

  

1. **Linear Model (Orange Line)**:

• This is a simple linear regression, meaning it’s a straight line trying to fit the data.

• Linear models are low in complexity and tend to underfit data when the true relationship is nonlinear. As you can see, the orange line doesn’t fit the data points very well but provides a relatively simple model.

2. **3rd Order Polynomial Model (Green Line)**:

• This is a 3rd-degree polynomial, allowing for a bit more flexibility than the linear model.

• It fits the data better than the linear model but is still relatively smooth.

• This model tries to balance between overfitting and underfitting, which is known as the **bias-variance trade-off**. It avoids too much complexity while still capturing some patterns in the data.

3. **10th Order Polynomial Model (Red Line)**:

• This is a 10th-degree polynomial, a highly flexible model.

• Notice how the red curve bends and twists to pass through most data points, showing that it is “overfitting” the data.

• Overfitting occurs when a model fits the training data too closely and captures noise or fluctuations that are not part of the underlying pattern. The red curve performs well within the training range but behaves poorly at the boundaries (for example, it spikes dramatically near 1.0), which suggests it will perform poorly on unseen data outside the training regime.

  

**Data and Model Complexity**

  

1. **Effect of Data**:

• **Noise**: The data points (blue circles) likely include random noise. In real-world datasets, noise is inevitable and can mislead a model into fitting random variations rather than capturing the true underlying pattern.

• **Sample Size**: The plot uses  data points, which is sufficient for training models, but the size and distribution of the data also influence how well models perform. If the data is noisy or sparse in certain regions, more complex models (like the 10th order polynomial) might fit that noise rather than the actual pattern.

2. **Effect of Model**:

• **Bias-Variance Trade-off**:

• A **linear model** has high bias (a strong assumption that the data follows a straight line) but low variance (it won’t change much with different data). It underfits the data.

• A **3rd-order polynomial** has moderate complexity, leading to a better balance between bias and variance. It fits the data better without being too flexible.

• A **10th-order polynomial** has low bias (it can fit almost anything) but very high variance. This model fits the training data closely, but as you can see from its erratic behavior outside the training region, it overfits the data, capturing noise rather than the actual pattern.

  

**Key Takeaways:**

  

• **Overfitting**: The 10th-order polynomial overfits the training data because it tries to capture every fluctuation (including noise). This results in poor generalization to new data, as shown by its erratic behavior outside the data range.

• **Underfitting**: The linear model underfits because it is too simple to capture the underlying pattern in the data. It fails to adjust to the variations in the data, as seen by its distance from many points.

• **Balanced Fit**: The 3rd-order polynomial strikes a balance between overfitting and underfitting. It captures the data trend well without trying to fit every minor fluctuation.

  

**How to Address This Behavior:**

  

• **Regularization**: Techniques like Lasso or Ridge regression can prevent overfitting in high-degree polynomial models by penalizing overly complex models.

• **Cross-validation**: Using cross-validation helps determine the optimal complexity for the model, ensuring it generalizes well to unseen data.

• **Model Selection**: It’s important to select a model based on the nature of the data. If you have a simple underlying relationship, a linear model may suffice. If there is more complexity but not too much noise, a polynomial model of moderate degree (like the 3rd-order polynomial) might be a good fit.

  

In conclusion, this plot exemplifies the common challenges in machine learning—finding the right model that captures the underlying data pattern without being too simple (underfitting) or too complex (overfitting).

![[Bias-Variance Trade-off.png]]

梯度下降法（Gradient Descent）的局限性。以下是详细的解释：

  

**1. 局部最小值问题**

  

• **梯度下降法找到局部最小值**：梯度下降算法可能会陷入局部最小值，尤其是在非凸损失函数的优化中。局部最小值是指函数在该点附近的值较小，但并非全局最小值。

• **引入随机性（stochasticity）**：为了解决这一问题，可以通过类似模拟退火（Simulated Annealing）的方法，引入一定的随机性。这种方法允许算法从局部最小值中跳出，有机会找到全局最小值。

  

**2. 对初始条件敏感**

  

• **初始条件敏感性**：梯度下降法的最终解往往受到参数初始值的影响，因为不同的初始点可能会导致不同的优化路径和结果。

• **解决方法**：需要仔细选择合理的初始条件，或者采用随机初始化的方法，增加找到全局最优解的机会。初始化策略在模型训练的成功中起着重要作用。

  

**3. 计算代价高**

  

• **大数据集上计算梯度代价高**：在处理大规模数据集时，计算每一步的梯度非常昂贵。尤其是全批量梯度下降（Batch Gradient Descent）每次更新参数时都需要用整个数据集来计算梯度，这会导致计算时间大大增加。

• **解决方法：使用小批量（Mini-batches）计算梯度**：可以使用随机梯度下降法（Stochastic Gradient Descent, SGD）或者小批量梯度下降法，使用数据集的子集来近似计算梯度。这样可以加速训练并引入一定的随机性，从而避免过早陷入局部最小值。

  

**小结：**

  

这张图片总结了梯度下降的局限性及应对策略，包括引入随机性来避免局部最优、对初始点的敏感性以及使用小批量数据来减小计算成本。

# 模型误差:

  
$$
\mathbb{E}[(\hat{y} - y)^2] = (\mathbb{E}[\hat{y}] - f(x))^2 \quad \text{(Bias)}^2 + \mathbb{E}[(\hat{y} - \mathbb{E}[\hat{y}])^2] \quad \text{(Variance)} + \sigma^2 \quad \text{(Irreducible error)}
$$



# 浅变量

**潜变量**（Latent Variables）是指**不可直接观测**但可以通过其他观测变量推断出的变量。在统计学、机器学习和数据科学中，潜变量通常用来表示隐藏的结构、潜在的过程或无法直接测量的抽象概念。

**潜变量的关键特征：**

1. **不可观测**：潜变量无法通过直接的测量手段得到，但可以通过其他观察到的变量间接推断。

2. **潜在因素**：它们通常代表影响观测数据的隐藏或潜在因素。

3. **在概率模型中常见**：潜变量常用于概率模型中，用来捕获数据的潜在结构或不确定性。

**潜变量的例子：**

• **心理学**：智力、幸福感、人格特质等都可以被视为潜变量，因为这些概念无法直接观察，但可以通过问卷或测试推断。

• **机器学习**：在主题模型（如潜在狄利克雷分配，LDA）中，“主题” 是潜变量，它们代表隐藏在文档中的主题，但并不能直接观测到，需要通过文档中的词推断出来。

• **经济学**：市场情绪、风险承受度等是潜变量，它们影响可观测的金融指标，但本身无法直接测量。

**潜变量模型的例子：**

1. **潜类模型（Latent Class Models）**：这种模型假设数据来自于不同的潜在类别（潜变量），但我们无法直接观测到这些类别，而是通过观察数据推断个体属于哪个类别。

2. **因子分析**：

• 因子分析利用潜变量来解释观测变量之间的相关性。例如，多个个体的行为可能受到潜在的特质如“智力”或“性格”的影响，而这些特质是潜变量。

3. **潜在狄利克雷分配 (LDA)**：

• LDA 是一种用于主题建模的模型，其中“主题”就是潜变量。通过文档中的词来推断出潜在的主题分布，而主题并不能直接观察到。

4. **隐马尔可夫模型 (HMM)**：

• 在 HMM 中，**隐藏状态**是潜变量。隐藏状态是我们无法直接观测到的，而是通过观测到的数据序列进行推断。

**概率图模型中的潜变量：**

在**混合模型**（如高斯混合模型，GMM）中，潜变量表示某个数据点是从哪个高斯分布中抽取出来的。我们无法直接观测到数据点属于哪个分布，只能通过模型推断。

例如，在 GMM 中，如果有两个高斯分布组成的混合模型，潜变量可以用来表示某个数据点属于第一个高斯分布还是第二个高斯分布，模型通过观察数据来推断出这个潜变量的值。

**在概率上下文中的正式定义：**

假设有一个概率模型，其中：

• x是观测到的数据。

• z是潜变量。

观测数据与潜变量的联合分布可以表示为：

P(X, Z) = P(X | Z) P(Z)

这里：  
• P(X | Z)表示给定潜变量时观测到数据  的概率。

• P(Z)是潜变量的先验分布。

由于潜变量  z 是未知的，我们需要通过积分将其边缘化，从而得到观测数据的边缘分布：
  

P(X) = \sum_{Z} P(X, Z) = \sum_{Z} P(X | Z) P(Z)

**潜变量的应用：**

1. **主题建模**：潜变量表示文档中的潜在主题。

2. **推荐系统**：用户偏好和物品特征等潜在因子是通过用户-物品交互数据推断出来的。

3. **聚类**：在高斯混合模型（GMM）中，潜变量表示数据点的簇归属。

4. **因果推断**：潜变量可以表示未测量的混杂因素，这些因素影响观测到的变量之间的关系。

**潜变量的估计：**

1. **期望最大化算法 (EM)**：这是用于估计潜变量的一种常见方法，尤其是在像 GMM 这样的概率模型中。EM 算法通过迭代地估计潜变量和模型参数来求解。

2. **变分推断**：这是一种用于逼近潜变量后验分布的技术，常用于复杂的概率模型如贝叶斯网络和 LDA。

3. **马尔可夫链蒙特卡洛 (MCMC)**：MCMC 方法用于从潜变量的后验分布中进行采样，常用于贝叶斯模型。

**总结：**

• **潜变量** 代表隐藏的、不可直接观测的因素，它们影响观测数据。

• 它们广泛应用于各种统计模型中，如因子分析、混合模型、隐马尔可夫模型和主题模型。

• 潜变量的估计通常通过算法如 EM、变分推断或 MCMC 来实现。

潜变量帮助我们捕捉数据中的隐藏结构，在许多实际问题中扮演着重要角色，尤其是当我们无法直接测量某些影响因素时。


# Supervised Learning
## def
a mapping from inputs to outputs

## parameters
the model is a family of possible relationships between inputs and outputs, the parameters specify the particular relationship

## Loss L
$$
L[ϕ]
$$
degree of mismatch
$$
ϕ = argmin ϕ [ L [ϕ] ] 
$$
parameters that minimize the loss

## Linear regression
### 1D linear regression model
$$
y = f[x, ϕ] = ϕ_0 + ϕ_1x.
$$
least-square loss:
$$L[\phi] = \sum_{i=1}^{I} (f[x_i, \phi] - y_i)^2 = \sum_{i=1}^{I} (\phi_0 + \phi_1 x_i - y_i)^2 .$$

### 1D Ising model
![[Pasted image 20241029121650.png]]
这张图展示了如何将一维Ising模型转化为线性回归问题。让我逐步解释各个要点：

1. 情境说明:

- 我们有一个自旋构型的系统集合
- 系统中的每个格点i都有一个自旋值si，可以是+1或-1
- 系统满足周期性边界条件
- 哈密顿量H表示系统的能量，由相邻自旋的相互作用决定

2. 目标:

- 建立一个模型来预测任意自旋构型的能量H

3. 理论假设:

- 采用成对相互作用的方式
- 每对自旋都可以相互作用
- 相互作用强度由参数Jij表示
- 系统的输入是自旋配置si，输出是能量H

4. 转化为线性回归:

- 将问题重写为标准的线性回归形式: y = X·w
- 其中:
    - y 代表能量H(s)
    - X 是由自旋积sisj构成的矩阵
    - w 包含相互作用参数Jij

这种方法的优点是可以将复杂的物理系统转化为标准的机器学习问题，使用线性回归的方法来学习系统的相互作用参数。这样就可以通过数据驱动的方式来理解和预测系统的行为。

$$H = -J\sum_{i=1}^{N} s_i s_{i+1}$$
$$H(\mathbf{s}) = \sum_{i,j} J_{ij} s_i s_j$$
$$y = X\mathbf{w}$$
其中：

- y 表示能量 H(s)
- X 是由自旋积 s_is_j 构成的设计矩阵
- w 是包含相互作用参数 J_{ij} 的权重向量

在这个模型中：

- s_i ∈ {+1, -1} 表示第i个位置的自旋状态
- J 是相邻自旋之间的相互作用强度
- N 是系统中的总自旋数
- 系统满足周期性边界条件，即 s_{N+1} = s_1

## Logistic Regression
### the logistic sigmoid
![[Pasted image 20241029124949.png]]
让我详细解释逻辑回归中Sigmoid函数的应用和意义：

1. 从线性回归到逻辑回归的转变
- 线性回归模型：s = xᵀw + b，其中s ∈ ℝ
- 这个模型输出连续值，不适合二分类问题
- 需要一个函数将连续输出映射到[0,1]区间，表示概率

2. 步进函数(Step Function)的问题
- 理想情况下，我们希望有一个清晰的决策边界
- 步进函数虽然可以实现二分类，但存在两个主要问题：
  * 不可微分（在跳跃点处导数不存在）
  * 变化太突兀（没有平滑过渡）

3. Logistic Sigmoid函数的引入
$$
\sigma(s) = \frac{1}{1 + e^{-s}}
$$
特点：
- 输出范围在(0,1)之间
- 处处可微
- S形曲线，有平滑的过渡
- 在原点处有最大梯度
- 两端渐近趋近于0和1

4. Sigmoid函数的优势
- 可微性：便于使用梯度下降等优化方法
- 概率解释：输出可以解释为属于类别1的概率
- 单调性：保持了原始特征的顺序关系
- 对称性：关于点(0,0.5)对称

5. 在二分类中的应用
- 当s → +∞时，σ(s) → 1，预测类别1
- 当s → -∞时，σ(s) → 0，预测类别0
- 决策边界在σ(s) = 0.5处，对应s = 0

6. 完整的逻辑回归模型
$$
P(y=1|x) = \sigma(x^Tw + b) = \frac{1}{1 + e^{-(x^Tw + b)}}
$$

7. 实际应用
- 在模型训练时：
  * 使用交叉熵损失函数
  * 通过梯度下降优化参数w和b
  * 可以添加正则化项防止过拟合
- 在预测时：
  * 如果σ(xᵀw + b) ≥ 0.5，预测类别1
  * 如果σ(xᵀw + b) < 0.5，预测类别0

Sigmoid函数的引入使得逻辑回归成为了一个强大的二分类模型，它不仅能给出类别预测，还能提供预测的概率，这在许多实际应用中都非常有用。

![[Pasted image 20241029125119.png]]

让我详细解释Logistic Sigmoid函数的重要性质：

1. 基本定义
$$
\sigma(s) = \frac{1}{1 + e^{-s}}
$$
这是Sigmoid函数的基本形式，将任意实数s映射到(0,1)区间。

2. 重要性质

a) 对称性质：
$$
1 - \sigma(s) = \sigma(-s)
$$
- 这表示Sigmoid函数关于点(0,0.5)对称
- 当s → +∞时，σ(s) → 1
- 当s → -∞时，σ(s) → 0
- 在s = 0处，σ(0) = 0.5

b) 导数性质：
$$
\sigma'(s) = \sigma(s)(1 - \sigma(s))
$$
这是一个非常优雅的性质：
- 导数可以用函数值本身表示
- 便于计算梯度
- 在反向传播中特别有用
- 导数最大值为0.25（在s=0处）

c) 导数的对称性：
$$
\sigma'(s) = \sigma'(-s)
$$
- 导数函数关于原点对称
- 这说明正负方向的变化率是相同的

3. 这些性质的重要应用

a) 在模型训练中：
- 导数性质使得梯度计算变得简单
- 对称性有助于模型的数值稳定性
- 导数有界，避免梯度爆炸

b) 在概率解释上：
- σ(s) + σ(-s) = 1，完美对应二分类的概率和为1
- 0到1的输出范围自然对应概率

4. 在Python中实现这些性质：

```python
import numpy as np

def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def sigmoid_derivative(s):
    sig = sigmoid(s)
    return sig * (1 - sig)

# 验证性质
s = 2.0
print(f"σ(s) = {sigmoid(s):.4f}")
print(f"1 - σ(s) = {1 - sigmoid(s):.4f}")
print(f"σ(-s) = {sigmoid(-s):.4f}")
print(f"σ'(s) = {sigmoid_derivative(s):.4f}")
```

5. 在深度学习中的应用

这些性质使得Sigmoid函数在多个方面都很有用：
- 二分类问题的输出层
- 门控循环神经网络（GRU, LSTM）中的门控机制
- 注意力机制中的权重计算

6. 优缺点分析

优点：
- 输出有明确的概率解释
- 导数形式简单
- 数值稳定性好
- 对称性好

缺点：
- 存在梯度消失问题（在饱和区）
- 输出不是零中心的
- 计算指数函数耗时
- 在深层网络中现已较少使用（常用ReLU代替）

7. 与其他激活函数的关系
- tanh(x) = 2σ(2x) - 1
- Softmax函数是Sigmoid的多维推广
- ReLU避免了Sigmoid的梯度消失问题

这些性质使得Logistic Sigmoid函数成为了机器学习中最重要的函数之一，尤其在二分类问题中。理解这些性质对于深入理解逻辑回归和神经网络都非常重要。

### 2-state statistical physics system

![[Pasted image 20241029125827.png]]

逻辑回归与统计物理中两态系统的联系：

1. 逻辑回归中的概率表示

对于数据点 xᵢ 和参数 θ：
$$
P(y_i=1|x_i,\theta) = \sigma(x_i^T\theta) = \frac{1}{1+e^{-x_i^T\theta}}
$$

对于类别0的概率：
$$
P(y_i=0|x_i,\theta) = 1 - P(y_i=1|x_i,\theta)
$$

2. 统计物理中的两态系统

系统能量表示：
$$
P(y_i=0) = \frac{e^{-\beta E_0}}{e^{-\beta E_0} + e^{-\beta E_1}} = \frac{1}{1+e^{-\beta(E_0-E_1)}}
$$

$$
P(y_i=1) = \frac{e^{-\beta E_1}}{e^{-\beta E_0} + e^{-\beta E_1}}
$$

其中：
- β = 1/(kT)，k是玻尔兹曼常数，T是温度
- E₀, E₁ 是系统的两个能量态
- ΔE = E₁ - E₀ 是能量差

3. 两者的对应关系

主要对应点：
- 逻辑回归中的 xᵢᵀθ 对应物理系统中的 -β(E₁-E₀)
- Sigmoid函数对应玻尔兹曼分布
- 分类边界对应能量简并点（E₀=E₁）

4. 物理意义的解释

a) 温度的作用：
- 高温（β小）：概率分布更均匀，对应模型的高不确定性
- 低温（β大）：概率分布更陡峭，对应模型的高确定性

b) 能量差的作用：
- 大的能量差：系统倾向于低能量态
- 小的能量差：两个态的概率接近

5. 在机器学习中的应用启示

a) 模型训练：
- 可以引入"退火"机制（类似模拟退火算法）
- 温度参数可以控制模型的确定性

b) 正则化理解：
- 可以从能量最小化的角度理解正则化
- L1/L2正则化对应不同的能量惩罚项

6. Python实现示例：

```python
import numpy as np

def logistic_probability(x, theta):
    """逻辑回归概率"""
    z = np.dot(x, theta)
    return 1 / (1 + np.exp(-z))

def boltzmann_probability(E0, E1, beta=1):
    """玻尔兹曼分布概率"""
    dE = E1 - E0
    return 1 / (1 + np.exp(beta * dE))
```

7. 实际应用的意义

这种联系帮助我们：
- 从物理角度理解机器学习
- 借鉴物理方法优化算法
- 设计新的学习策略
- 理解模型的不确定性

这种物理学与机器学习的联系不仅有理论意义，也为算法优化和模型设计提供了新的思路。例如，我们可以：
- 使用物理系统的模拟方法来优化模型
- 借鉴统计力学的系综理论来分析模型的不确定性
- 利用温度参数来控制模型的探索与利用平衡

### Loss function

逻辑回归中的损失函数构造过程，特别是通过最大似然估计(MLE)的推导：

1. 最大似然估计的基本思想
- 目标是找到使观测数据概率最大的参数θ
- 对数似然被用来简化计算

2. 观测数据的概率表达式：

对单个数据点的概率：
$$P(y_i|x_i,\theta) = (\sigma(x_i^T\theta))^{y_i}(1-\sigma(x_i^T\theta))^{1-y_i}$$

这里：
- yi = 1时，得到$$P(y_i=1|x_i,\theta) = \sigma(x_i^T\theta)$$
- yi = 0时，得到$$P(y_i=0|x_i,\theta) = 1-\sigma(x_i^T\theta)$$

3. 最大似然估计（MLE）的形式化表达：

$$\hat{\theta} = \arg\max_{\theta} \log(P(Y|X,\theta))$$

4. 假设数据点独立，可以将联合概率分解：

$$\hat{\theta} = \arg\max_{\theta} \sum_{i=1}^n \log(P(y^i|x^i,\theta))$$

5. 展开对数似然：

$$\ell(\theta) = \sum_{i=1}^n [y_i\log(\sigma(x_i^T\theta)) + (1-y_i)\log(1-\sigma(x_i^T\theta))]$$

6. 损失函数（Cost function）定义：
$$C(\theta) = -\ell(\theta)$$

$$C(\theta) = -\sum_{i=1}^n [y_i\log(\sigma(x_i^T\theta)) + (1-y_i)\log(1-\sigma(x_i^T\theta))]$$

这就是著名的交叉熵损失函数。

7. 理解这个损失函数：

a) 对于正例(yi = 1)：
- 损失项为$$-\log(\sigma(x_i^T\theta))$$
- 当预测概率接近1时，损失趋近于0
- 当预测概率接近0时，损失趋近于无穷大

b) 对于负例(yi = 0)：
- 损失项为$$-\log(1-\sigma(x_i^T\theta))$$
- 当预测概率接近0时，损失趋近于0
- 当预测概率接近1时，损失趋近于无穷大

8. 优化目标：
$$\theta^* = \arg\min_{\theta} C(\theta)$$

这个优化问题的特点：
- 凸优化问题，保证全局最优解
- 可以使用梯度下降方法求解
- 损失函数处处可导

9. 梯度计算：

对于单个样本的梯度：
$$\frac{\partial C}{\partial \theta} = (h_\theta(x_i) - y_i)x_i$$

其中$$h_\theta(x_i) = \sigma(x_i^T\theta)$$

10. 实际应用中的优化策略：

```python
def log_likelihood(X, y, theta):
    """计算对数似然"""
    return np.sum(
        y * np.log(sigmoid(X @ theta)) + 
        (1 - y) * np.log(1 - sigmoid(X @ theta))
    )

def gradient_descent(X, y, theta, learning_rate, n_iterations):
    """梯度下降优化"""
    for _ in range(n_iterations):
        gradient = X.T @ (sigmoid(X @ theta) - y)
        theta = theta - learning_rate * gradient
    return theta
```

11. 正则化扩展：

可以添加正则化项来防止过拟合：

L2正则化的损失函数：
$$C(\theta) = -\ell(\theta) + \lambda\|\theta\|_2^2$$

L1正则化的损失函数：
$$C(\theta) = -\ell(\theta) + \lambda\|\theta\|_1$$

这种通过最大似然估计推导出的损失函数具有以下优点：
- 理论基础扎实（基于统计学原理）
- 优化性质好（凸函数）
- 概率解释清晰
- 易于实现和优化

### Optimizing loss function

让我详细解释如何优化逻辑回归的损失函数：

1. 损失函数回顾
损失函数（Cost function）定义为：
$$C(\theta) = -\sum_{i=1}^n [y_i\log(\sigma(x_i^T\theta)) + (1-y_i)\log(1-\sigma(x_i^T\theta))]$$

2. 梯度下降优化
通过计算损失函数对参数θ的梯度：
$$\nabla_\theta C(\theta) = \sum_{i=1}^n (\sigma(x_i^T\theta) - y_i)x_i$$

这个梯度有几个重要特点：
- 形式简洁
- 计算高效
- 直观含义明确：预测值与真实值的差乘以特征向量

3. 梯度推导过程：

a) 首先利用链式法则：
$$\frac{\partial C}{\partial \theta} = \frac{\partial C}{\partial \sigma} \cdot \frac{\partial \sigma}{\partial \theta}$$

b) 考虑sigmoid函数的导数：
$$\frac{\partial \sigma(z)}{\partial z} = \sigma(z)(1-\sigma(z))$$

c) 最终得到梯度表达式：
$$\nabla_\theta C(\theta) = \sum_{i=1}^n (\sigma(x_i^T\theta) - y_i)x_i$$

4. 梯度下降更新规则：
$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta C(\theta_t)$$
其中：
- α是学习率
- t表示迭代次数

5. Python实现：

```python
def logistic_regression_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    # 初始化参数
    theta = np.zeros(X.shape[1])
    
    for _ in range(n_iterations):
        # 计算预测值
        z = np.dot(X, theta)
        h = 1 / (1 + np.exp(-z))
        
        # 计算梯度
        gradient = np.dot(X.T, (h - y))
        
        # 更新参数
        theta -= learning_rate * gradient
    
    return theta
```

6. 优化策略：

a) 学习率选择：
- 太大可能导致发散
- 太小收敛太慢
- 常用范围：0.001 ~ 0.1

b) 批量处理：
- 批量梯度下降（BGD）：使用所有样本
- 随机梯度下降（SGD）：每次使用单个样本
- 小批量梯度下降（Mini-batch GD）：使用部分样本

c) 正则化：
添加正则化项的梯度：
$$\nabla_\theta C(\theta) = \sum_{i=1}^n (\sigma(x_i^T\theta) - y_i)x_i + \lambda\theta$$

7. 终止条件：

可以使用以下条件之一：
- 达到最大迭代次数
- 梯度范数小于阈值
- 损失函数变化小于阈值

8. 实现细节：

```python
def train_logistic_regression(X, y, learning_rate=0.01, n_iterations=1000, tolerance=1e-6):
    theta = np.zeros(X.shape[1])
    prev_cost = float('inf')
    
    for i in range(n_iterations):
        # 预测和梯度计算
        z = np.dot(X, theta)
        predictions = 1 / (1 + np.exp(-z))
        gradient = np.dot(X.T, (predictions - y)) / len(y)
        
        # 更新参数
        theta -= learning_rate * gradient
        
        # 计算当前损失
        current_cost = compute_cost(X, y, theta)
        
        # 检查收敛
        if abs(prev_cost - current_cost) < tolerance:
            print(f"Converged after {i} iterations")
            break
            
        prev_cost = current_cost
    
    return theta
```

此优化方法的主要优点是：
- 实现简单
- 计算效率高
- 可以处理大规模数据
- 收敛性有保证（因为损失函数是凸函数）

需要注意的问题：
- 学习率的选择
- 特征的预处理和标准化
- 正则化参数的调整
- 收敛条件的设置

-------

## Training
The process of finding the optimal parameters

# Testing
Computing the loss on a separate set of test data.
Depending on how complete and representative the training data + how expressive the model is.


