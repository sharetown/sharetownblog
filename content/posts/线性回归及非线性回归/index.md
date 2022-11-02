---
title: "线性回归及非线性回归"
description: 但是
date: 2022-10-26T22:14:25+08:00
draft: false

categories:
- python

tags:
- 机器学习
- 线性回归
- 非线性回归
---
{{<katex>}} 
线性还是非线性是要根据分析的目标来决定的，在线性回归和非线性回归中，我们需要求解的是模型参数，因而，线性与非线性描述的是函数模型与模型参数之间的关系，而非因变量与自变量之间的关系


## 代价函数（损失函数）（Cost function）
- 最小二乘法
- 真实值\\(y\\),预测值\\(h_\theta\\),则误差平方为\\((y-h_\theta(x))^2\\).
- 找到合适的参数，使得误差平方和：

对于线性的：
$$
h_\theta(x)=\theta_0+\theta_1x
\\\\
J(\theta_0,\theta_1)=\dfrac{1}{2m}\textstyle\sum_{i=1}^m(y^i-h_\theta(x^j))^2
$$
最小

- 我们使用相关系数去衡量线性相关的强弱：
$$
r_{xy}=\dfrac{\sum(X_i-\overline{X})(Y_i-\overline{Y})}{\sqrt{\sum(X_i-\overline{X})^2\sum(Y_i-\overline{Y})^2}}
$$
其中\\(X_i\\)表示真实值的横坐标；\\(Y_i\\)表示真实值纵横坐标；\\(\overline{X}\\)表示真实值的横坐标的平均值；\\(\overline{Y}\\)表示真实值的纵坐标的平均值。
- 相关系数\\(R^2\\)是用来描述两个变量之间的线性关系的，但决定系数的适用范围更广, 可以用于描述非线性或者有个及两个以上自变量的相关关系。它可以用来评价模型的效果。
- 总平方和（SST）：\\(\textstyle\sum_{i=1}^n(y_i-\overline{y})^2\\)
- 回归平方和（SSR）：\\(\textstyle\sum_{i=1}^n(\hat{y}-\overline{y})^2\\)
- 残差平方和（SSE）：\\(\textstyle\sum_{i=1}^n(y_i-\hat{y})^2\\)
- 它们三者的关系是：\\(SST=SSR+SSE\\)
- 决定系数：\\(R^2=\dfrac{SSR}{SST}=1-\dfrac{SSE}{SST}\\)

## 一、梯度下降
需要做一个迭代：
$$
\theta_j:=\theta_j-\alpha\dfrac{\partial}{\partial\theta_j}J(\theta_0,\theta_1),j=0,1
$$
其中\\(\alpha\\)为学习率；\\(:=\\)为赋值符，将右边赋值给左边

这个迭代公式在更新的时候必须同步更新，即：
$$
temp0:=\theta_0-\alpha\dfrac{\partial}{\partial\theta_0}J(\theta_0,\theta_1)
\\\\
temp1:=\theta_1-\alpha\dfrac{\partial}{\partial\theta_1}J(\theta_0,\theta_1)
\\\\
\theta_0:=temp0
\\\\
\theta_1:=temp1
$$
学习率不能太小也不能太大。太小计算耗时，太大就会发生震荡。

### 1、用梯度下降法来求解线性回归
上面提到的\\(\alpha\dfrac{\partial}{\partial\theta_j}J(\theta_0,\theta_1)\\)拆开来是这样的：
$$
j=0:\alpha\dfrac{\partial}{\partial\theta_j}J(\theta_0,\theta_1)=\dfrac{1}{m}\displaystyle\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})
$$
$$
j=1:\alpha\dfrac{\partial}{\partial\theta_j}J(\theta_0,\theta_1)=\dfrac{1}{m}\displaystyle\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})\cdotp x^{(i)}
$$
需要注意的是用梯度下降法来求解线性回归，如果函数是凸函数那么最终可以求得全局最优解，但如果函数是非凸函数，那么求得的解就有可能陷入局部最优。
#### 实战1.1 一元线性回归
**语言**：Python

**一、使用numpy实现:**

**第三方库**：numpy、matplotlib。

```python
import numpy as np
import matplotlib.pyplot as plt
#----------------------上面是导包
# 载入数据
data = np.genfromtxt("data.csv", delimiter=",")
# 注意data.csv的路径
x_data = data[:,0]
y_data = data[:,1]
plt.scatter(x_data,y_data)
plt.show()
#-----------------------上面是载入数据后给原始数据利用matplotlib画出原始数据的散点图
# 学习率learning rate
lr = 0.0001
# 截距
b = 0 
# 斜率
k = 0 
# 最大迭代次数
epochs = 50
#------------------------上面这段初始化学习率、截距、斜率和最大迭代次数
# 最小二乘法
def compute_error(b, k, x_data, y_data):
    totalError = 0
    for i in range(0, len(x_data)):
        totalError += (y_data[i] - (k * x_data[i] + b)) ** 2
    return totalError / float(len(x_data)) / 2.0
#--------------------------上面这段是利用最小二乘法求误差平方和
def gradient_descent_runner(x_data, y_data, b, k, lr, epochs):#这个函数就是利用梯度下降迭代来更新斜率和截距
    # 计算总数据量
    m = float(len(x_data))
    # 循环epochs次
    for i in range(epochs):
        b_grad = 0#临时截距
        k_grad = 0#临时斜率
        # 计算梯度的总和再求平均
        for j in range(0, len(x_data)):
            b_grad += (1/m) * (((k * x_data[j]) + b) - y_data[j])
            k_grad += (1/m) * x_data[j] * (((k * x_data[j]) + b) - y_data[j])
        # 更新b和k
        b = b - (lr * b_grad)
        k = k - (lr * k_grad)
        #---------------------完全按照梯度下降的公式实现的。
        # 每迭代5次，输出一次图像
#         if i % 5==0:
#             print("epochs:",i)
#             plt.plot(x_data, y_data, 'b.')
#             plt.plot(x_data, k*x_data + b, 'r')
#             plt.show()
#被注释这一段可以每五次输出一张迭代后的图，可以很直观的看到对比。
    return b, k

print("Starting b = {0}, k = {1}, error = {2}".format(b, k, compute_error(b, k, x_data, y_data)))
print("Running...")
b, k = gradient_descent_runner(x_data, y_data, b, k, lr, epochs)
print("After {0} iterations b = {1}, k = {2}, error = {3}".format(epochs, b, k, compute_error(b, k, x_data, y_data)))

#画图
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, k*x_data + b, 'r')
plt.show()
#-----------------最后这一段就是一个结果的输出
```
#### 实战1.2 直接用sklearn库实现：（记住这个）
**第三方库**：numpy、matplotlib、sklearn。
```python
from sklearn.linear_model import LinearRegression#导入sklearn中处理线性回归的包LinearRegression
import numpy as np
import matplotlib.pyplot as plt
#----------------------导包
# 载入数据
data = np.genfromtxt("data.csv", delimiter=",")
x_data = data[:,0]
y_data = data[:,1]
plt.scatter(x_data,y_data)
plt.show()
print(x_data.shape)
#----------------------导入数据并画出原始数据的散点图
x_data = data[:,0,np.newaxis]#将数据的0列的元素切片出来，并利用np.newaxis这个参数使其具有维度。
y_data = data[:,1,np.newaxis]
# 创建并拟合模型
model = LinearRegression()#创建LinearRegression类的对象model
model.fit(x_data, y_data)#model对象调用fit函数，将两列元素传进去，由已经封装好的函数进行计算
#--------------------------用sklearn的LinearRegression进行建模
# 画图
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, model.predict(x_data), 'r')
plt.show()
#-------------------------只画图
```

#### 多元线性回归
当\\(y\\)的影响因素不是唯一时，采用多元线性回归模型。

**多特征**
$$
h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2+\dots+\theta_nx_n
$$
那么在这里的话有多少个特征就会有多少个\\(x\\)。

**多元线性回归的梯度下降**
$$
\theta_j:=\theta_j-\alpha\dfrac{1}{m}\displaystyle\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)}) x_j^{(i)}
\\\\
(j=0,\dots,n)
$$
### 2、梯度下降法解决多元线性回归
**实战2.1：利用numpy使用梯度下降法**
```python
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D#画3D图的库
#----------------------------导库
data = genfromtxt(r"Delivery.csv",delimiter=',')
print(data)
# 切分数据
x_data = data[:,:-1]
y_data = data[:,-1]
print(x_data)
print(y_data)
#-----------------------------导入数据

# 学习率learning rate
lr = 0.0001
# 参数
theta0 = 0
theta1 = 0
theta2 = 0
# 最大迭代次数
epochs = 1000
# 最小二乘法
def compute_error(theta0, theta1, theta2, x_data, y_data):
    totalError = 0
    for i in range(0, len(x_data)):
        totalError += (y_data[i] - (theta1 * x_data[i,0] + theta2*x_data[i,1] + theta0)) ** 2
    return totalError / float(len(x_data))
#-----------------------------------求代价函数

def gradient_descent_runner(x_data, y_data, theta0, theta1, theta2, lr, epochs):
    # 计算总数据量
    m = float(len(x_data))
    # 循环epochs次
    for i in range(epochs):
        theta0_grad = 0
        theta1_grad = 0
        theta2_grad = 0
        # 计算梯度的总和再求平均
        for j in range(0, len(x_data)):
            theta0_grad += (1/m) * ((theta1 * x_data[j,0] + theta2*x_data[j,1] + theta0) - y_data[j])
            theta1_grad += (1/m) * x_data[j,0] * ((theta1 * x_data[j,0] + theta2*x_data[j,1] + theta0) - y_data[j])
            theta2_grad += (1/m) * x_data[j,1] * ((theta1 * x_data[j,0] + theta2*x_data[j,1] + theta0) - y_data[j])
        # 更新b和k
        theta0 = theta0 - (lr*theta0_grad)
        theta1 = theta1 - (lr*theta1_grad)
        theta2 = theta2 - (lr*theta2_grad)
    return theta0, theta1, theta2
#--------------------------------梯度下降法迭代更新参数

print("Starting theta0 = {0}, theta1 = {1}, theta2 = {2}, error = {3}".
      format(theta0, theta1, theta2, compute_error(theta0, theta1, theta2, x_data, y_data)))
print("Running...")
theta0, theta1, theta2 = gradient_descent_runner(x_data, y_data, theta0, theta1, theta2, lr, epochs)
print("After {0} iterations theta0 = {1}, theta1 = {2}, theta2 = {3}, error = {4}".
      format(epochs, theta0, theta1, theta2, compute_error(theta0, theta1, theta2, x_data, y_data)))
      
#------------------------------调用写的函数并输出结果
ax = plt.figure().add_subplot(111, projection = '3d') 
ax.scatter(x_data[:,0], x_data[:,1], y_data, c = 'r', marker = 'o', s = 100) #点为红色三角形  
x0 = x_data[:,0]
x1 = x_data[:,1]
# 生成网格矩阵
x0, x1 = np.meshgrid(x0, x1)
z = theta0 + x0*theta1 + x1*theta2
# 画3D图
ax.plot_surface(x0, x1, z)
#设置坐标轴  
ax.set_xlabel('Miles')  
ax.set_ylabel('Num of Deliveries')  
ax.set_zlabel('Time')  
  
#显示图像  
plt.show() 
#----------------------------画图，并且是画3D图

```
**实战2.2：利用sklearn库实现多元线性回归（代码少）**
```python
import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  
#------------------------------导库
# 读入数据 
data = genfromtxt(r"Delivery.csv",delimiter=',')
print(data)
# 切分数据
x_data = data[:,:-1]
y_data = data[:,-1]
print(x_data)
print(y_data)
#-----------------------------导入数据
# 创建模型
model = linear_model.LinearRegression()
model.fit(x_data, y_data)
#----------------------------利用库建模
# 系数
print("coefficients:",model.coef_)

# 截距
print("intercept:",model.intercept_)

# 测试
x_test = [[102,4]]
predict = model.predict(x_test)
print("predict:",predict)
#-------------------------------输出计算出的模型的参数并进行测试
ax = plt.figure().add_subplot(111, projection = '3d') 
ax.scatter(x_data[:,0], x_data[:,1], y_data, c = 'r', marker = 'o', s = 100) #点为红色三角形  
x0 = x_data[:,0]
x1 = x_data[:,1]
# 生成网格矩阵
x0, x1 = np.meshgrid(x0, x1)
z = model.intercept_ + x0*model.coef_[0] + x1*model.coef_[1]
# 画3D图
ax.plot_surface(x0, x1, z)
#设置坐标轴  
ax.set_xlabel('Miles')  
ax.set_ylabel('Num of Deliveries')  
ax.set_zlabel('Time')  
  
#显示图像  
plt.show()  
#------------------------------画3D图
```
### 3、多项式回归
多项式回归，回归函数是回归变量多项式的回归。多项式回归模型是线性回归模型的一种，此时回归函数关于回归系数是线性的。由于任一函数都可以用多项式逼近，因此多项式回归有着广泛应用。
$$
y=\theta_0+\theta_1x+\theta_2x+\theta_3x+\dots+\theta_nx
$$
**实战**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
#--------------------------导包
# 载入数据
data = np.genfromtxt("job.csv", delimiter=",")
x_data = data[1:,1]
y_data = data[1:,2]
plt.scatter(x_data,y_data)
plt.show()
x_data = x_data[:,np.newaxis]
y_data = y_data[:,np.newaxis]
#---------------------------数据的导入
# 创建并拟合模型
model = LinearRegression()
model.fit(x_data, y_data)
#------------------------------先使用一元线性回归的模型来拟合
# 画图
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, model.predict(x_data), 'r')
plt.show()
#-------------------------------画出使用一元线性回归来拟合多项式的结果
# 定义多项式回归,degree的值可以调节多项式的特征
poly_reg  = PolynomialFeatures(degree=5) 
# 特征处理
x_poly = poly_reg.fit_transform(x_data)
# 定义回归模型
lin_reg = LinearRegression()
# 训练模型
lin_reg.fit(x_poly, y_data)
#---------------------------------定义多项式回归并训练模型
# 画图
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, lin_reg.predict(poly_reg.fit_transform(x_data)), c='r')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#--------------------------------画出使用多项式回归的结果图
```
## 二、标准方程法
标准方程法就是令式子的代价函数的偏导都等于0时所求得的解向量就是使代价函数最小的解。
$$
令\dfrac{\partial}{\partial\theta_j}J(\theta)=\dots=0
\\\\
求解:\theta_0,\theta_1,\dots,\theta_n
$$
我们知道，在梯度下降法中，式子的代价函数为\\(J(\theta_0,\theta_1,\dots,\theta_n)=\dfrac{1}{2m}\textstyle\sum_{i=1}^m(y^i-h_\theta(x^j))^2\\)
当我们有一组数据，我们把自变量用矩阵\\(X\\)表示，自变量用矩阵\\(y\\)表示，需要求解的令代价函数等于零的解用矩阵\\(w\\)表示。例如：
$$
X=\begin{bmatrix}x_{0,0}&x_{0,1}&x_{0,2}&x_{0,3}&x_{0,4} \\\\ 
x_{1,0}&x_{1,1}&x_{1,2}&x_{1,3}&x_{1,4} \\\\
x_{2,0}&x_{2,1}&x_{2,2}&x_{2,3}&x_{2,4} \\\\
x_{3,0}&x_{3,1}&x_{3,2}&x_{3,3}&x_{3,4}\end{bmatrix}
\\\\
w=\begin{bmatrix}w_0 \\\\ w_1 \\\\ w_2 \\\\ w_3 \\\\ w_4 \end{bmatrix}
\\\\
y=\begin{bmatrix}y_0 \\\\ y_1 \\\\ y_2 \\\\ y_3 \\\\ y_4 \end{bmatrix}
\\\\
\displaystyle\sum_{i=1}^m(h_w(x^i)-y_i)^2=(y-Xw)^T(y-Xw)
$$
### 这里涉及到矩阵求导
**分子布局：** 分子为列向量或者分母为行向量 

**分母布局：** 分子为行向量或者分母为列向量
$$
\dfrac{\partial(y-Xw)^T(y-Xw)}{\partial w}
\\\\
\dfrac{\partial(y^Ty-y^TXw-w^TX^Ty+w^TX^TXw)}{\partial w}
\\\\
\dfrac{\partial y^Ty}{\partial w}-\dfrac{\partial y^TXw}{\partial w}-\dfrac{\partial w^TX^Ty}{\partial w}+\dfrac{\partial w^TX^TXw}{\partial w}
$$
**矩阵的求导百度查表**

在这里我们查表后可以求得：
$$
\dfrac{\partial y^Ty}{\partial w}=0
\\\\
\dfrac{\partial y^TXw}{\partial w}=X^Ty
\\\\
\dfrac{\partial w^TX^Ty}{\partial w}=\dfrac{\partial(w^TX^Ty)^T}{\partial w}=\dfrac{\partial y^TXw}{\partial w}=X^Ty
\\\\
\dfrac{\partial w^TX^TXw}{\partial w}=2X^TXw
$$
那么：
$$
\dfrac{\partial y^Ty}{\partial w}-\dfrac{\partial y^TXw}{\partial w}-\dfrac{\partial w^TX^Ty}{\partial w}+\dfrac{\partial w^TX^TXw}{\partial w}=0-X^Ty-X^Ty+2X^TXw
\\\\
-2X^Ty+2X^TXw=0
\\\\
X^TXw=X^Ty
\\\\
(X^TX)^{-1}X^TXw=(X^TX)^{-1}X^Ty
\\\\
w=(X^TX)^{-}X^Ty
$$
**矩阵不可逆的情况**
1、线性相关的特征(多重共线性)。
例如:\\(x_1\\)为房子的面积,单位是平方英尺
\\(x_2\\)为房子的面积,单位是平方米
预测房价
1平方英尺\\(≈0.0929\\)平方米
2、特征数据太多(样本数m≤特征数量n )

### 梯度下降法和标准方程法的优缺点对比
方程类型|优点|缺点
--|--|--
梯度下降法|当特征值非常多的时候也可以很好的工作|需要选择合适的学习率<br>需要迭代很多个周期<br>只能得到最优解的近似值
标准方程法|不需要学习率<br>不需要迭代<br>可以得到全局最优解|需要计算`$(X^TX)^{-1}$`<br>时间复杂度大约是`$O(n^3)$`<br>n是特征数量
**实战：** 标准方程法解决线性回归
```python
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
#-----------------------------------
# 载入数据
data = np.genfromtxt("data.csv", delimiter=",")
x_data = data[:,0,np.newaxis]
y_data = data[:,1,np.newaxis]
plt.scatter(x_data,y_data)
plt.show()
print(np.mat(x_data).shape)
print(np.mat(y_data).shape)
# 给样本添加偏置项
X_data = np.concatenate((np.ones((100,1)),x_data),axis=1)
print(X_data.shape)
#----------------------------------
# 标准方程法求解回归参数
def weights(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xTx = xMat.T*xMat # 矩阵乘法
    # 计算矩阵的值,如果值为0，说明该矩阵没有逆矩阵
    if np.linalg.det(xTx) == 0.0:
        print("This matrix cannot do inverse")
        return
    # xTx.I为xTx的逆矩阵
    ws = xTx.I*xMat.T*yMat
    return ws
#----------------------------------
#调用上面这个函数
ws = weights(X_data,y_data)
#----------------------------------
    # 画图
x_test = np.array([[20],[80]])
y_test = ws[0] + x_test*ws[1]
plt.plot(x_data, y_data, 'b.')
plt.plot(x_test, y_test, 'r')
plt.show()
```
## 三、特征缩放
当数据中的特征的值相差较大时将不利于我们拟合以及作图。例如\\(x_1=房子的面积（1000000cm^2-2000000cm^2)\\)
\\(x_2=房间的数量（1-5）\\)
### 1、数据归一化
数据归一化就是把数据的取值范围处为0-1或者-1-1之间
 任意数据转化为0-1之间：\\(newValue=(oldValue-min)/(max-min)\\)
  任意数据转化为-1-1之间：\\(newValue=((oldValue-min)/(max-min)-0.5)*2\\)
### 2、均值标准化
\\(newValue=(oldValue-u)/s\\)<br>x为特诊数据，u为数据的平均值，s为数据的方差
## 四、交叉验证法
当数据中的记录数量本身不是很多时，如果将数据集按原来的方式划分成训练集和测试集时，就会降低我们训练的效果。这时我们可以将数据划分成n等份（一般划分为10份）然后需要循环n次，每一次循环都取不一样的一份为测试机，剩下的全部为训练集，这样每执行一次循环都会得到一个误差值E，那么最后我们所有误差求和后在求平均值就可以得到较好的误差的值\\(E=\dfrac{1}{n}\displaystyle\sum_{i=1}^nE_i\\)
## 五、过拟合及正则化
拟合情况一般可以分为三种：欠拟合、正确拟合和过拟合。
**欠拟合**：就是拟合程度不够导致模型在训练集和测试集中都表现较差。
**正确拟合**：模型拟合程度高，在训练集和测试集里的表现都很好
**过拟合**：模型的拟合程度太极端，导致模型在训练集中的表现非常完美，但是在测试集中往往会出现很多误差
### 防止过拟合的措施
1.减少特征
2.增加数据量
3.正则化
### 正则化
正则化代价函数与普通的代价函数差不多，唯一的不同就是最后会加一项\\(\lambda\displaystyle\sum_{j=1}^n\theta_j^2\\)或则\\(\lambda\displaystyle\sum_{j=1}^n\vert\theta_j\vert\\)。前者叫L2正则化，后者叫L1正则化。
## 六、岭回归
在前面标准方程法中求得的权值的表达式是：\\(w=(X^TX)^{-}X^Ty\\)。如果数据的特征比样本点还多, 数据特征n,样本个数m，如果n> m,则计算\\((X^TX)^{-1}\\)时会出错。因为\\((X^TX)\\)不是满秩矩阵，所以不可逆。<br>
为了解决这个问题,统计学家引入了岭回归的概念。
$$
w=(X^T+\lambda I)^{-1}X^Ty
$$
\\(\lambda\\)为岭系数, \\(I\\)为单位矩阵(对角线.上全为1 ,其他元素全为0)
岭回归的代价函数是之前谈到的L2正则化。
$$
J(\theta)=\dfrac{1}{2}\displaystyle\sum_{i=1}^n(h_\theta(x_i)-y_i)^2+\lambda\displaystyle\sum_{i}^n\theta_i^2
$$
岭回归最早是用来处理特征数多于样本的情况,现在也
用于在估计中加入偏差,从而得到更好的估计。同时也
可以解决多重共线性的问题。岭回归是一种有偏估计。
**岭回归代价函数：** \\(J(\theta)=\dfrac{1}{2m}\left[\displaystyle\sum_{i=1}^m(h_\theta(x_i)-y_i)^2+\lambda\displaystyle\sum_{j}^n\theta_j^2\right]\\)
**线性回归标准方程法：** \\(w=(X^TX)^{-}X^Ty\\)
**岭回归求解：** \\(w=(X^T+\lambda I)^{-1}X^Ty.\lambda为岭系数\\)
在选择\\(\lambda\\)的值的时候，需要考查下面两个问题使得:
1.各回归系数的岭估计基本稳定。
2.残差平方和增大不太多。